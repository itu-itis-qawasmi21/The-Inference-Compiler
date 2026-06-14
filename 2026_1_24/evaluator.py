import tensorflow as tf
import numpy as np
import time

class PerformanceEvaluator:
    def __init__(self, original_model, optimized_model, test_data):
        self.orig_model = original_model
        self.opt_model = optimized_model
        self.test_x, self.test_y = test_data

    def _ensure_compiled(self, model):
        """
        FIXED: Ensures model is compiled before evaluate() is called.
        """
        # Checks if 'compiled' flag is set. If not, compiles.
        if not getattr(model, 'compiled', False):
            try:
                model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
            except Exception:
                # Fallback for older versions/custom objects
                model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )

    def _count_params(self, model):
        """Returns total vs non-zero parameters."""
        total = 0
        non_zero = 0
        for layer in model.layers:
            weights = layer.get_weights()
            for w in weights:
                total += w.size
                non_zero += np.count_nonzero(w)
        return total, non_zero

    def _measure_latency(self, model, iterations=50):
        """Measures average inference time per batch in ms."""
        # Warmup
        _ = model(self.test_x[:1], training=False)
        
        start_time = time.time()
        for _ in range(iterations):
            _ = model(self.test_x[:32], training=False) # Batch 32
        end_time = time.time()
        
        avg_time = (end_time - start_time) / iterations
        return avg_time * 1000 # Convert to ms

    def _calculate_theoretical_flops(self, model):
        """
        Estimates Multiply-Accumulate (MAC) operations.
        Accounts for Sparsity in the optimized model.
        """
        total_flops = 0
        active_flops = 0
        
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                w, b = layer.get_weights()
                input_dim = w.shape[0]
                output_dim = w.shape[1]
                
                layer_flops = 2 * input_dim * output_dim
                non_zeros = np.count_nonzero(w)
                active_layer_flops = (2 * non_zeros) + b.size 
                
                total_flops += layer_flops
                active_flops += active_layer_flops

            elif isinstance(layer, tf.keras.layers.Conv2D):
                weights = layer.get_weights()
                if len(weights) == 2:
                    w, b = weights
                else:
                    w = weights[0]
                    # No bias
                
                # --- ROBUST SHAPE RETRIEVAL (Keras 3 Compatible) ---
                h_out, w_out = 32, 32 # Default fallback for CIFAR
                
                try:
                    # Attempt 1: Standard Keras Functional API
                    output_shape = layer.output.shape
                    if output_shape[1] is not None: h_out = output_shape[1]
                    if output_shape[2] is not None: w_out = output_shape[2]
                except Exception:
                    try:
                        # Attempt 2: Layer method (for multi-output layers)
                        output_shape = layer.get_output_shape_at(0)
                        if output_shape[1] is not None: h_out = output_shape[1]
                        if output_shape[2] is not None: w_out = output_shape[2]
                    except Exception:
                        # Attempt 3: Legacy/Internal (Keras 2)
                        try:
                            output_shape = layer.output_shape
                            if output_shape[1] is not None: h_out = output_shape[1]
                            if output_shape[2] is not None: w_out = output_shape[2]
                        except Exception:
                            # Fallback already set to 32x32
                            pass

                k_h, k_w, c_in, c_out = w.shape
                ops_per_filter = k_h * k_w * c_in
                layer_flops = 2 * h_out * w_out * ops_per_filter * c_out
                
                density = np.count_nonzero(w) / w.size
                active_layer_flops = layer_flops * density
                
                total_flops += layer_flops
                active_flops += active_layer_flops
                
        return total_flops, active_flops

    def evaluate(self):
        print("\n" + "="*50)
        print("    FINAL PERFORMANCE EVALUATION REPORT")
        print("==================================================")
        
        self._ensure_compiled(self.orig_model)
        self._ensure_compiled(self.opt_model)

        # 1. ACCURACY
        print("\n[1] ACCURACY METRICS")
        _, acc_orig = self.orig_model.evaluate(self.test_x, self.test_y, verbose=0)
        _, acc_opt = self.opt_model.evaluate(self.test_x, self.test_y, verbose=0)
        drop = acc_orig - acc_opt
        
        print(f"  Original Model:   {acc_orig*100:.2f}%")
        print(f"  Optimized Model:  {acc_opt*100:.2f}%")
        print(f"  Accuracy Drop:    {drop*100:.2f}%  [{'PASS' if drop < 0.02 else 'WARNING'}]")

        # 2. SPARSITY & PARAMETERS
        print("\n[2] SPARSITY METRICS")
        total_p, active_p = self._count_params(self.opt_model)
        sparsity = 1.0 - (active_p / max(1, total_p))
        print(f"  Total Params:      {total_p:,}")
        print(f"  Active Params:     {active_p:,}")
        print(f"  Global Sparsity:  {sparsity*100:.2f}%")

        # 3. THEORETICAL COMPUTATION (FLOPs)
        print("\n[3] COMPUTATIONAL COST (Approx. FLOPs)")
        flops_orig, flops_opt = self._calculate_theoretical_flops(self.opt_model)
        
        print(f"  Dense FLOPs:       {int(flops_orig):,}")
        print(f"  Sparse FLOPs:      {int(flops_opt):,}")
        if flops_opt > 0:
            print(f"  Theoretical Speedup: {flops_orig / flops_opt:.2f}x")

        # 4. LATENCY (Python Baseline)
        print("\n[4] PYTHON INFERENCE LATENCY (Batch 32)")
        lat_orig = self._measure_latency(self.orig_model)
        lat_opt = self._measure_latency(self.opt_model)
        print(f"  Original:          {lat_orig:.2f} ms")
        print(f"  Optimized:         {lat_opt:.2f} ms")
        print("  (Note: Real speedup will be higher in generated C++)")

        # 5. LAYER BREAKDOWN
        print("\n[5] LAYER-WISE BREAKDOWN")
        print(f"  {'Layer Name':<20} | {'Shape':<15} | {'Sparsity':<10} | {'Unique Vals'}")
        print("-" * 65)
        for layer in self.opt_model.layers:
            if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
                w = layer.get_weights()[0]
                sp = 1.0 - (np.count_nonzero(w) / w.size)
                unique = len(np.unique(w))
                shape_str = str(w.shape)
                print(f"  {layer.name:<20} | {shape_str:<15} | {sp*100:.1f}%      | {unique}")

        # Verdict calculation
        success = (drop < 0.02) and (sparsity > 0.5)
        print("\n" + "="*50)
        print(f"OVERALL VERDICT: {'SUCCESS [PASS]' if success else 'NEEDS TUNING [WARNING]'}")
        print("==================================================\n")
        
        return {
            "accuracy_original": acc_orig,
            "accuracy_optimized": acc_opt,
            "sparsity": sparsity,
            "theoretical_speedup": flops_orig / (flops_opt + 1e-9)
        }