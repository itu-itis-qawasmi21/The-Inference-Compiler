import tensorflow as tf
import numpy as np
import math
import copy
import gc
import keras
import sys  # Added for buffer flushing
from keras import backend as K

class PruningEngine:
    """
    Handles the removal of neurons (Structured) and weights (Unstructured).
    
    OPTIMIZED STRATEGY:
    1. Phase 1 (Greedy): Uses "Smart Ranking" + "Proxy Data" to find candidates fast, 
       then commits them using a Safe Batch Backoff on the full dataset.
    2. Phase 2 (Heuristic): Uses dynamic sensitivity analysis with recovery logic.
    3. Phase 3 (Weights): Uses adaptive thresholding for fine-grained pruning.
    """
    def __init__(self, model, compile_config):
        self.model = model
        self.compile_config = compile_config
        self.step_counter = 0
        
        # --- COMPATIBILITY: Support Dense (MLP) and Conv2D (ResNet) ---
        self.PRUNABLE_LAYERS = (tf.keras.layers.Dense, tf.keras.layers.Conv2D)
        
        # Baselines
        self.initial_accuracy = 0.0
        self.best_accuracy_seen = 0.0 
        self.greedy_peak_accuracy = 0.0 
        
        # State Tracking
        self.dead_neurons = {}  
        self.dead_weights = {}  
        self.protected_neurons = {} # Tracks neurons that caused crashes (with cooldown)
        self.last_weights_cache = {}
        
        # Switch Logic State
        self.consecutive_protected_count = 0
        
        # --- LIMITS (Based on your values) ---
        self.NEURON_TOTAL_LIMIT = 0.013 
        self.NEURON_STEP_LIMIT = 0.002
        self.TOTAL_DROP_LIMIT = 0.02          
        self.GREEDY_BATCH_DROP_LIMIT = 0.015 # Max drop allowed for a greedy batch

        # Metric Weights (Activation / Sensitivity / Edge)
        self.ALPHA = 0.2 
        self.BETA = 0.6  
        self.GAMMA = 0.2 

    def _cache_weights(self):
        """Backs up current weights for restoration."""
        self.last_weights_cache = {}
        for layer in self.model.layers:
            if isinstance(layer, self.PRUNABLE_LAYERS):
                self.last_weights_cache[layer.name] = layer.get_weights()

    def _restore_weights(self):
        """Restores weights from cache."""
        for layer_name, weights in self.last_weights_cache.items():
            layer = self.model.get_layer(layer_name)
            layer.set_weights(weights)

    def _update_baseline(self, acc):
        if acc > self.best_accuracy_seen:
            self.best_accuracy_seen = acc

    def _enforce_constraints(self):
        """
        Re-applies masks to ensure dead neurons/weights stay dead.
        Includes ResNet Fix: If Conv2D filter is killed, kill the following BN channel too.
        """
        layers = self.model.layers
        for i, layer in enumerate(layers):
            if not isinstance(layer, self.PRUNABLE_LAYERS): continue
            
            weights_list = layer.get_weights()
            if len(weights_list) == 2:
                w, b = weights_list
            else:
                w = weights_list[0]
                b = None

            # 1. STRUCTURED MASKING (Neurons/Filters)
            if layer.name in self.dead_neurons:
                indices = list(self.dead_neurons[layer.name])
                if indices:
                    # Kill Filter/Neuron
                    w[..., indices] = 0.0
                    if b is not None:
                        b[indices] = 0.0
                    
                    # --- RESNET FIX: Handle BatchNormalization ---
                    if (i + 1) < len(layers):
                        next_layer = layers[i+1]
                        if isinstance(next_layer, tf.keras.layers.BatchNormalization):
                            bn_weights = next_layer.get_weights()
                            bn_weights[0][indices] = 0.0 # Gamma
                            bn_weights[1][indices] = 0.0 # Beta
                            bn_weights[2][indices] = 0.0 # Mean
                            bn_weights[3][indices] = 0.0 # Var
                            next_layer.set_weights(bn_weights)
            
            # 2. UNSTRUCTURED MASKING (Weights)
            if layer.name in self.dead_weights:
                w = w * self.dead_weights[layer.name]
            
            if b is not None:
                layer.set_weights([w, b])
            else:
                layer.set_weights([w])

    def _get_global_sparsity(self):
        """Calculates current model sparsity."""
        total_p = 0
        active_p = 0
        for layer in self.model.layers:
            if isinstance(layer, self.PRUNABLE_LAYERS):
                weights_list = layer.get_weights()
                w = weights_list[0]
                total_p += w.size
                mask = np.ones_like(w)
                
                if layer.name in self.dead_weights:
                    mask = mask * self.dead_weights[layer.name]
                
                if layer.name in self.dead_neurons:
                    mask[..., list(self.dead_neurons[layer.name])] = 0.0
                
                active_p += np.sum(mask)
        return 1.0 - (active_p / total_p) if total_p > 0 else 0

    def _calculate_all_neuron_scores(self, train_data):
        """Calculates importance scores for all neurons (Rank Metric)."""
        target_layers = [l for l in self.model.layers if isinstance(l, self.PRUNABLE_LAYERS)]
        grad_model = tf.keras.Model(inputs=self.model.inputs, outputs=[l.output for l in target_layers])
        
        # Use one batch for ranking (Fast approximation)
        try:
            batch_images, batch_labels = next(iter(train_data))
        except:
            print("  ! Warning: Could not fetch training batch. Using zeros for gradient score.")
            batch_images = None
        
        grads = [None] * len(target_layers)
        outputs = [None] * len(target_layers)

        if batch_images is not None:
            with tf.GradientTape() as tape:
                outputs = grad_model(batch_images)
                if len(target_layers) == 1: outputs = [outputs]
                loss = self.compile_config['loss_fn'](batch_labels, outputs[-1])
            grads = tape.gradient(loss, outputs)
        
        all_scores = []
        for i, layer in enumerate(target_layers):
            if outputs[i] is None: 
                out_shape = layer.output_shape
                if isinstance(out_shape, list): out_shape = out_shape[0]
                if len(out_shape) == 4:
                    outputs[i] = np.zeros((1, out_shape[1], out_shape[2], out_shape[3]))
                else:
                    outputs[i] = np.zeros((1, out_shape[1]))

            axes_to_reduce = tuple(range(len(outputs[i].shape) - 1))
            
            if grads[i] is not None:
                sens = np.mean(np.abs(grads[i].numpy()), axis=axes_to_reduce)
            else:
                sens = np.zeros(outputs[i].shape[-1])
            
            act = np.mean(np.abs(outputs[i].numpy()), axis=axes_to_reduce)
            
            w = layer.get_weights()[0]
            w_axes = tuple(range(len(w.shape) - 1))
            col_sums = np.sum(np.abs(w), axis=w_axes) + 1e-9
            edge = np.max(np.abs(w) / col_sums, axis=w_axes)
            
            def z_score(v): 
                return (v - np.mean(v)) / (np.std(v) + 1e-9)
            
            layer_scores = (self.ALPHA * z_score(act)) + \
                           (self.BETA * z_score(sens)) + \
                           (self.GAMMA * z_score(edge))
            
            if layer.name not in self.dead_neurons: self.dead_neurons[layer.name] = set()
            
            for idx, score in enumerate(layer_scores):
                if idx not in self.dead_neurons[layer.name]:
                    is_protected = False
                    if layer.name in self.protected_neurons:
                        if idx in self.protected_neurons[layer.name]:
                            if self.step_counter < self.protected_neurons[layer.name][idx]:
                                is_protected = True
                            else:
                                del self.protected_neurons[layer.name][idx]
                    if not is_protected:
                        all_scores.append((score, layer.name, idx))
        return all_scores

    # --- OPTIMIZATION: Stratified Sampling for Proxy Data ---
    def _get_proxy_data(self, x, y, size=1000):
        """Creates a balanced 1,000-image subset for fast scanning."""
        indices = []
        y_int = np.argmax(y, axis=1) if len(y.shape) > 1 else y
        classes = np.unique(y_int)
        
        per_class = max(1, size // len(classes))
        
        for cls in classes:
            cls_idx = np.where(y_int == cls)[0]
            selected = cls_idx[:per_class]
            indices.extend(selected)
        
        indices = np.array(indices)
        np.random.shuffle(indices)
        
        return x[indices], y[indices]

    # =========================================================================
    # PHASE 1: RANKED GREEDY SWEEP (Optimized)
    # =========================================================================
    def _run_greedy_sweep_phase(self, train_data, val_data):
        print(f"\n--- PHASE 1: RANKED GREEDY SWEEP (Rank -> Proxy Scan -> Safe Batch) ---")
        full_x, full_y = val_data
        
        # 1. OPTIMIZATION: Use Proxy Data for the "Individual Scan"
        proxy_x, proxy_y = self._get_proxy_data(full_x, full_y, size=1000)
        
        # 2. RANKING: Calculate scores to find "Least Value Nodes"
        print("  > Ranking neurons by Importance (Metric Calculation)...")
        sys.stdout.flush()
        all_scores = self._calculate_all_neuron_scores(train_data)
        
        # Sort: Lowest score = Most useless
        all_scores.sort(key=lambda x: x[0])
        
        # Target: Check the bottom 15% of filters
        check_limit = int(len(all_scores) * 0.15)
        candidates_to_check = all_scores[:check_limit]
        print(f"  > Scanning bottom {check_limit} candidates on Proxy Set...")

        candidates = []
        
        # 3. PROXY SCAN: Verify individual safety
        for i, (score, lname, idx) in enumerate(candidates_to_check):
            if i % 50 == 0: print(f"    > Scanned {i}/{check_limit}...", end="\r")
            
            # Fast Check: Prune Temporarily
            layer = self.model.get_layer(lname)
            w_list = layer.get_weights()
            w = w_list[0]
            b = w_list[1] if len(w_list) > 1 else None
            
            orig_col = w[..., idx].copy()
            w[..., idx] = 0.0
            if b is not None: b[idx] = 0.0
            
            layer.set_weights([w, b] if b is not None else [w])
            
            # Test on PROXY (Fast)
            _, acc = self.model.evaluate(proxy_x, proxy_y, verbose=0, batch_size=1000)
            
            # Restore
            w[..., idx] = orig_col
            layer.set_weights([w, b] if b is not None else [w])
            
            # If it passes Proxy (within 0.1% tolerance), add to batch candidates
            if acc >= (self.best_accuracy_seen - 0.001):
                candidates.append((acc, lname, idx))

        print(f"\n  > Candidates passed proxy check: {len(candidates)}")
        if not candidates:
            self.greedy_peak_accuracy = self.best_accuracy_seen
            return

        # Sort candidates: Highest accuracy (least damage) first
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # 4. SAFE BATCH COMMIT (Binary Search Backoff)
        print(f"  > Attempting Safe Batch Pruning on FULL Dataset...")
        
        current_batch = candidates
        while current_batch:
            self._cache_weights()
            
            # Prepare batch for pruning
            temp_dead = copy.deepcopy(self.dead_neurons)
            for _, lname, idx in current_batch:
                if lname not in temp_dead: temp_dead[lname] = set()
                temp_dead[lname].add(idx)
            
            # Apply masks temporarily
            for layer in self.model.layers:
                if layer.name in temp_dead:
                    w_list = layer.get_weights()
                    w = w_list[0]
                    b = w_list[1] if len(w_list) > 1 else None
                    
                    indices = list(temp_dead[layer.name])
                    if indices: 
                        w[..., indices] = 0.0
                        if b is not None: b[indices] = 0.0
                    
                    layer.set_weights([w, b] if b is not None else [w])
            
            # TEST ON FULL DATA
            _, batch_acc = self.model.evaluate(full_x, full_y, verbose=0, batch_size=4096)
            drop = self.best_accuracy_seen - batch_acc
            
            if drop > self.GREEDY_BATCH_DROP_LIMIT:
                print(f"    > Batch of {len(current_batch)} Failed (Drop {drop*100:.2f}%). Backing off...")
                self._restore_weights()
                
                # BACKOFF: Cut batch size in half
                new_size = len(current_batch) // 2
                if new_size < 1:
                    print("    > No safe batch found. Stopping Greedy Phase.")
                    break
                current_batch = current_batch[:new_size]
            else:
                # Success! Commit.
                self.dead_neurons = temp_dead
                self._update_baseline(batch_acc)
                print(f"    > Success! Pruned {len(current_batch)} neurons. New Acc: {batch_acc:.4f}")
                
                # Enforce constraints properly (handles BN, Weights, etc.)
                self._enforce_constraints()
                break # One-Shot logic complete

    # =========================================================================
    # PHASE 2: DYNAMIC HEURISTIC
    # =========================================================================
    def _handle_heuristic_failure(self, current_victims, val_data):
        print(f"    > Safety Check Failed. Analyzing victims for guilty neurons...")
        self._restore_weights()
        
        flat_victims = []
        for lname, idxs in current_victims.items():
            for idx in idxs: flat_victims.append((lname, idx))
            
        guilty, innocent = [], []
        
        # Use subset for quick blame assignment
        subset_x = val_data[0][:1000]
        subset_y = val_data[1][:1000]
        
        for lname, idx in flat_victims:
            self.dead_neurons[lname].add(idx)
            self._enforce_constraints()
            
            # Quick Check
            _, acc = self.model.evaluate(subset_x, subset_y, verbose=0)
            drop = self.best_accuracy_seen - acc
            
            if (drop > (self.NEURON_STEP_LIMIT * 1.5)):
                guilty.append((lname, idx))
            else:
                innocent.append((lname, idx))
                
            self.dead_neurons[lname].discard(idx)
            self._restore_weights()

        new_deadline = self.step_counter + 15
        for lname, idx in guilty:
            if lname not in self.protected_neurons: self.protected_neurons[lname] = {}
            self.protected_neurons[lname][idx] = new_deadline
            
        # Refresh existing protections
        refreshed_count = 0
        for lname in self.protected_neurons:
            for idx in self.protected_neurons[lname]:
                if self.protected_neurons[lname][idx] > self.step_counter:
                    self.protected_neurons[lname][idx] = new_deadline
                    refreshed_count += 1
        
        # Innocent can be pruned
        for lname, idx in innocent: self.dead_neurons[lname].add(idx) 
        self._enforce_constraints()
        
        print(f"    > Recovery: Protected {len(guilty)} new + {refreshed_count} existing. Pruned {len(innocent)} innocent.")
        return len(guilty)

    def _run_dynamic_heuristic_phase(self, train_data, val_data):
        print(f"\n--- PHASE 2: DYNAMIC HEURISTIC (Step Limit 0.2% | Total Limit 1.3%) ---")
        sys.stdout.flush()
        
        while True:
            self.step_counter += 1
            
            # Check Protection Limits
            active_prot = 0
            for lname in self.protected_neurons:
                active_prot += sum(1 for t in self.protected_neurons[lname].values() if t > self.step_counter)
            
            if active_prot >= 30 or (active_prot > 10 and self.consecutive_protected_count >= 5):
                print(f"  > Resistance Detected (Active Prot: {active_prot}). Switching to Weights.")
                break

            self._cache_weights()
            all_scores = self._calculate_all_neuron_scores(train_data)
            if not all_scores: break
            all_scores.sort(key=lambda x: x[0])
            
            # Prune 0.5% at a time
            n_prune = max(1, int(len(all_scores) * 0.005))
            victims = all_scores[:n_prune]
            
            current_batch = {}
            for _, lname, idx in victims:
                if lname not in current_batch: current_batch[lname] = []
                current_batch[lname].append(idx)
            
            for lname, idxs in current_batch.items():
                for idx in idxs: self.dead_neurons[lname].add(idx)
            self._enforce_constraints()
            
            # --- [SPEED FIX] Limit Fine-Tuning ---
            print(f"  > Step {self.step_counter}: Fine-tuning (50 batches)...")
            self.model.fit(train_data, steps_per_epoch=50, epochs=1, verbose=1)
            self._enforce_constraints()
            
            _, acc = self.model.evaluate(val_data[0], val_data[1], verbose=0)
            self._update_baseline(acc)
            drop = self.best_accuracy_seen - acc
            
            step_fail = (drop > (self.NEURON_STEP_LIMIT + 0.0005))
            budget_fail = (acc < (self.greedy_peak_accuracy - self.NEURON_TOTAL_LIMIT))
            
            if step_fail or budget_fail: 
                # Revert and Analyze
                for lname, idxs in current_batch.items():
                    for idx in idxs: self.dead_neurons[lname].discard(idx)
                n_prot = self._handle_heuristic_failure(current_batch, val_data)
                if n_prot > 0: self.consecutive_protected_count += n_prot
            else:
                self.consecutive_protected_count = 0
                curr_sp = self._get_global_sparsity()
                print(f"  > Success. Acc: {acc:.4f} Base: {self.best_accuracy_seen:.4f} (Sparsity: {curr_sp*100:.1f}%)")
                sys.stdout.flush()

    # =========================================================================
    # PHASE 3: WEIGHT PRUNING
    # =========================================================================
    def _run_weight_phase(self, train_data, val_data):
        print(f"\n--- PHASE 3: WEIGHT PRUNING ---")
        
        for layer in self.model.layers:
            if isinstance(layer, self.PRUNABLE_LAYERS):
                w = layer.get_weights()[0]
                if layer.name not in self.dead_weights:
                    self.dead_weights[layer.name] = np.ones_like(w)

        # 1. ADAPTIVE THRESHOLD SWEEP
        factor = 2.0
        print(f"  > Starting Adaptive Sweep (Factor {factor} -> 0.3)...")
        
        while factor >= 0.3:
            self._cache_weights()
            mask_backup = copy.deepcopy(self.dead_weights)
            
            total_pruned = 0
            for layer in self.model.layers:
                if isinstance(layer, self.PRUNABLE_LAYERS):
                    w = layer.get_weights()[0]
                    fan_in = np.prod(w.shape[:-1]) 
                    threshold = factor / (fan_in + 10.0)
                    
                    w_axes = tuple(range(len(w.shape) - 1))
                    col_sums = np.sum(np.abs(w), axis=w_axes) + 1e-9
                    rel_importance = np.abs(w) / col_sums
                    
                    mask = (rel_importance < threshold) & (self.dead_weights[layer.name] == 1.0)
                    if layer.name in self.dead_neurons:
                        dead_cols = list(self.dead_neurons[layer.name])
                        mask[..., dead_cols] = False
                    
                    count = np.sum(mask)
                    if count > 0:
                        self.dead_weights[layer.name][mask] = 0.0
                        total_pruned += count
            
            if total_pruned == 0:
                factor = round(factor - 0.2, 1)
                continue

            self._enforce_constraints()
            self.model.fit(train_data, steps_per_epoch=50, epochs=1, verbose=0)
            self._enforce_constraints()
            
            _, acc = self.model.evaluate(val_data[0], val_data[1], verbose=0)
            self._update_baseline(acc)
            
            total_drift = self.best_accuracy_seen - acc 
            
            if total_drift > self.TOTAL_DROP_LIMIT:
                print(f"    - Factor {factor:.1f}: Too Aggressive (Drop {total_drift*100:.2f}%). Reverting.")
                self._restore_weights()
                self.dead_weights = mask_backup
                factor = round(factor - 0.2, 1) 
            else:
                print(f"    - Factor {factor:.1f}: Success! (Acc {acc:.4f}). Rejecting and applying buffer.")
                self._restore_weights()
                self.dead_weights = mask_backup
                
                safe_factor = max(0.3, factor - 0.2)
                print(f"    > Applying Safer Factor {safe_factor:.1f}...")
                
                # Apply Safe Factor
                for layer in self.model.layers:
                    if isinstance(layer, self.PRUNABLE_LAYERS):
                        w = layer.get_weights()[0]
                        fan_in = np.prod(w.shape[:-1])
                        threshold = safe_factor / (fan_in + 10.0)
                        
                        w_axes = tuple(range(len(w.shape) - 1))
                        col_sums = np.sum(np.abs(w), axis=w_axes) + 1e-9
                        rel_importance = np.abs(w) / col_sums
                        
                        mask = (rel_importance < threshold) & (self.dead_weights[layer.name] == 1.0)
                        if layer.name in self.dead_neurons:
                            mask[..., list(self.dead_neurons[layer.name])] = False
                        
                        if np.sum(mask) > 0:
                            self.dead_weights[layer.name][mask] = 0.0
                            
                self._enforce_constraints()
                self.model.fit(train_data, steps_per_epoch=50, epochs=1, verbose=0)
                self._enforce_constraints()
                
                _, final_acc = self.model.evaluate(val_data[0], val_data[1], verbose=0)
                self._update_baseline(final_acc)
                curr_sp = self._get_global_sparsity()
                print(f"  > Sweep Complete. Acc: {final_acc:.4f} Base: {self.best_accuracy_seen:.4f} (Sparsity: {curr_sp*100:.1f}%)")
                break 

        # 2. ITERATIVE HEURISTIC WEIGHT PRUNING
        print(f"  > Starting Iterative Weight Pruning...")
        
        while True:
            self.step_counter += 1
            self._cache_weights()
            mask_backup = copy.deepcopy(self.dead_weights)
            
            target_layers = [l for l in self.model.layers if isinstance(l, self.PRUNABLE_LAYERS)]
            grad_model = tf.keras.Model(inputs=self.model.inputs, outputs=[l.output for l in target_layers])
            
            try:
                batch_images, batch_labels = next(iter(train_data))
            except:
                break

            with tf.GradientTape() as tape:
                outputs = grad_model(batch_images)
                if len(target_layers) == 1: outputs = [outputs]
                loss = self.compile_config['loss_fn'](batch_labels, outputs[-1])
            grads = tape.gradient(loss, outputs)
            
            all_w_scores = []
            for i, layer in enumerate(target_layers):
                w = layer.get_weights()[0]
                axes = tuple(range(len(outputs[i].shape) - 1))
                
                if grads[i] is not None:
                    sens = np.mean(np.abs(grads[i].numpy()), axis=axes) 
                else:
                    sens = np.zeros(outputs[i].shape[-1])
                    
                act = np.mean(np.abs(outputs[i].numpy()), axis=axes)
                
                broadcast_shape = [1] * (len(w.shape) - 1) + [w.shape[-1]]
                score_mat = np.abs(w) * (1.0 + (self.BETA * sens.reshape(broadcast_shape)) + (self.ALPHA * act.reshape(broadcast_shape)))
                
                # Ignore already dead
                score_mat[self.dead_weights[layer.name] == 0] = np.inf
                if layer.name in self.dead_neurons:
                    score_mat[..., list(self.dead_neurons[layer.name])] = np.inf
                
                flat = score_mat.flatten()
                valid = np.where(flat != np.inf)[0]
                for idx in valid: all_w_scores.append((flat[idx], layer.name, idx))
            
            if not all_w_scores: break
            all_w_scores.sort(key=lambda x: x[0])
            
            n_prune = max(3000, int(len(all_w_scores) * 0.04)) 
            victims = all_w_scores[:n_prune]
            
            for _, lname, fidx in victims:
                l = self.model.get_layer(lname)
                w_shape = l.get_weights()[0].shape
                coords = np.unravel_index(fidx, w_shape)
                self.dead_weights[lname][coords] = 0.0
            
            self._enforce_constraints()
            self.model.fit(train_data, steps_per_epoch=50, epochs=1, verbose=0)
            self._enforce_constraints()
            
            _, acc = self.model.evaluate(val_data[0], val_data[1], verbose=0)
            self._update_baseline(acc)
            
            total_drift = self.best_accuracy_seen - acc 
            
            if total_drift > self.TOTAL_DROP_LIMIT:
                print(f"  !!! Limit Hit: Total Drift {total_drift*100:.2f}% > {self.TOTAL_DROP_LIMIT*100}%. Stopping.")
                self._restore_weights()            
                self.dead_weights = mask_backup    
                break
            
            curr_sp = self._get_global_sparsity()
            print(f"  Step {self.step_counter}: Pruned weight batch. Acc: {acc:.4f} (Sparsity: {curr_sp*100:.1f}%)")

    def run_iterative_pruning(self, train_data, val_data, target_sparsity=None):
        print(f"--- [Pruning Engine] Optimized Hybrid Mode ---")
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
                           loss=self.compile_config['loss_fn'], metrics=['accuracy'])
        
        _, self.initial_accuracy = self.model.evaluate(val_data[0], val_data[1], verbose=0)
        self.best_accuracy_seen = self.initial_accuracy
        print(f"  Initial Accuracy: {self.initial_accuracy:.4f}")

        # 1. Optimized Greedy (Rank + Proxy + Batch Backoff)
        self._run_greedy_sweep_phase(train_data, val_data)
        
        # 2. Dynamic Heuristic (Safety Recovery)
        self._run_dynamic_heuristic_phase(train_data, val_data)
        
        # 3. Weights (Fine-tuning)
        self._run_weight_phase(train_data, val_data)
        
        return self.model