import tensorflow as tf
import numpy as np

class QuantizationEngine:
    def __init__(self, model):
        self.model = model
        self.QUANTIZABLE_LAYERS = (tf.keras.layers.Dense, tf.keras.layers.Conv2D)

    def _get_proxy_data(self, x, y, size=1000):
        """
        Creates a balanced subset (10 samples per class) for fast sensitivity checks.
        """
        indices = []
        y_int = np.argmax(y, axis=1) if len(y.shape) > 1 else y
        classes = np.unique(y_int)
        per_class = max(1, size // len(classes))

        for cls in classes:
            idx = np.where(y_int == cls)[0][:per_class]
            indices.extend(idx)

        return x[indices], y[indices]

    def _fake_quantize_weights(self, weights, mode='linear'):
        """
        Simulates 8-bit quantization scaling and rounding.
        """
        min_val = np.min(weights)
        max_val = np.max(weights)
        abs_max = max(abs(min_val), abs(max_val))

        if abs_max == 0:
            return weights

        # Scale to Int8 range [-127, 127]
        scale = 127.0 / abs_max
        quantized = np.round(weights * scale)

        # De-quantize back to Float32 for simulation
        return quantized / scale

    def _safe_get_weights(self, layer):
        """
        Safely unpacks layer weights regardless of whether a bias exists.
        ResNet Conv2D layers pruned without bias return only [w].
        Returns (w, b) where b may be None.
        """
        weights_list = layer.get_weights()
        w = weights_list[0]
        b = weights_list[1] if len(weights_list) > 1 else None
        return w, b

    def analyze_sensitivity(self, val_x, val_y, mode='linear', drop_tolerance=0.02):
        """
        Greedy Layer-Wise Check:
        1. Quantizes ONE layer.
        2. Checks accuracy on PROXY data.
        3. If drop > tolerance, marks layer as SENSITIVE (do not quantize).
        """
        sensitivity_map = {}
        print(f"\n[Quantization] Running Layer-Wise Sensitivity Check (Proxy Mode)...")

        # 1. Generate Proxy Data (10x Speedup)
        proxy_x, proxy_y = self._get_proxy_data(val_x, val_y, size=1000)

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # 2. Get Baseline on Proxy
        _, baseline_acc = self.model.evaluate(proxy_x, proxy_y, verbose=0)
        print(f"  Proxy Baseline Acc: {baseline_acc:.4f}")

        # 3. Iterate Groups (Layers)
        for layer in self.model.layers:
            if not isinstance(layer, self.QUANTIZABLE_LAYERS):
                continue

            # Backup — safe unpack handles bias-free Conv2D layers
            w_orig, b_orig = self._safe_get_weights(layer)

            # Apply Test Quantization
            w_quant = self._fake_quantize_weights(w_orig, mode=mode)
            if b_orig is not None:
                layer.set_weights([w_quant, b_orig])
            else:
                layer.set_weights([w_quant])

            # Fast Inference Check
            _, acc = self.model.evaluate(proxy_x, proxy_y, verbose=0)

            # Decision Logic
            drop = baseline_acc - acc
            if drop > drop_tolerance:
                print(f"  [Sensitive] {layer.name}: Drop {drop*100:.2f}% -> MARKED TO SKIP")
                sensitivity_map[layer.name] = False  # False = Do Not Quantize
            else:
                sensitivity_map[layer.name] = True   # True = Safe to Quantize

            # Restore Weights (Critical for isolating layer effect)
            if b_orig is not None:
                layer.set_weights([w_orig, b_orig])
            else:
                layer.set_weights([w_orig])

        return sensitivity_map

    def apply_qat(self, sensitivity_map, accuracy_threshold, mode='linear'):
        """
        Applies quantization ONLY to groups (layers) marked as Safe.
        """
        print(f"\n[Quantization] Applying Conditional Quantization...")

        quantized_count = 0
        skipped_count = 0

        for layer in self.model.layers:
            if not isinstance(layer, self.QUANTIZABLE_LAYERS):
                continue

            # CHECK: Should we skip this group?
            if layer.name in sensitivity_map and sensitivity_map[layer.name] == False:
                skipped_count += 1
                continue

            # Safe unpack handles bias-free Conv2D layers
            w, b = self._safe_get_weights(layer)

            # 1. Round Weights
            w_quant = self._fake_quantize_weights(w)

            # 2. Identity Snap (Optimization)
            # If weight is 1.0 (within error), snap it to 1.0 to skip multiplication
            mask = (np.abs(w_quant - 1.0) < 0.01)
            if np.sum(mask) > 0:
                w_quant[mask] = 1.0

            if b is not None:
                layer.set_weights([w_quant, b])
            else:
                layer.set_weights([w_quant])

            quantized_count += 1

        print(f"  > Result: Quantized {quantized_count} layers. Skipped {skipped_count} sensitive layers.")
        return self.model
