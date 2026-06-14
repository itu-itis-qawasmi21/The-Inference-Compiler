import torch
import numpy as np
import tensorflow as tf
from keras import layers, models

def build_wrn_40_2_fused(input_shape=(32, 32, 3), num_classes=100):
    """
    Defines WideResNet-40-2 with FUSED Input Normalization.
    IMPLEMENTS XTERNALZ PRE-ACTIVATION LOGIC CORRECTLY.
    """
    depth = 40
    widen_factor = 2
    nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
    n = (depth - 4) // 6
    
    def basic_block(x, in_planes, out_planes, stride, stage, block, dropRate=0.0):
        prefix = f"layer{stage}.{block}"
        
        # 1. Capture Raw Input (for Identity shortcut case)
        raw_x = x
        
        # 2. Shared BN1 + ReLU1
        # In xternalz, if dimensions change, this is applied to x BEFORE branching.
        x = layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=f"{prefix}.bn1")(x)
        x = layers.Activation('relu', name=f"{prefix}.relu1")(x)
        
        # 3. Capture Activated Input (for ConvShortcut case)
        activated_x = x
        
        # --- Main Path ---
        # Conv1 takes Activated Input
        x = layers.Conv2D(out_planes, 3, strides=stride, padding='same', use_bias=False, 
                          name=f"{prefix}.conv1")(x)
        
        # BN2 + ReLU2
        x = layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=f"{prefix}.bn2")(x)
        x = layers.Activation('relu', name=f"{prefix}.relu2")(x)
        
        if dropRate > 0:
            x = layers.Dropout(dropRate, name=f"{prefix}.dropout")(x)
            
        # Conv2
        x = layers.Conv2D(out_planes, 3, strides=1, padding='same', use_bias=False, 
                          name=f"{prefix}.conv2")(x)
        
        # --- Shortcut Logic ---
        if stride != 1 or in_planes != out_planes:
            # Case A: Dimensions Change (e.g. Block1.0, Block2.0)
            # CRITICAL FIX: Shortcut takes ACTIVATED x, not raw x.
            shortcut = layers.Conv2D(out_planes, 1, strides=stride, padding='same', use_bias=False,
                                     name=f"{prefix}.downsample.0")(activated_x)
        else:
            # Case B: Identity (Dimensions match)
            # Shortcut takes RAW x (skipping the first BN/ReLU of this block)
            shortcut = raw_x
            
        return layers.Add(name=f"{prefix}.add")([x, shortcut])

    # --- Model Entry ---
    inputs = layers.Input(shape=input_shape)
    
    # Initial Conv (Fused Bias Enabled)
    x = layers.Conv2D(nChannels[0], 3, padding='same', use_bias=True, name="conv1")(inputs)
    
    # Block 1
    for i in range(n):
        stride = 1 if i == 0 else 1
        in_planes = nChannels[0] if i == 0 else nChannels[1]
        x = basic_block(x, in_planes, nChannels[1], stride, stage=1, block=i)
        
    # Block 2
    for i in range(n):
        stride = 2 if i == 0 else 1
        in_planes = nChannels[1] if i == 0 else nChannels[2]
        x = basic_block(x, in_planes, nChannels[2], stride, stage=2, block=i)
        
    # Block 3
    for i in range(n):
        stride = 2 if i == 0 else 1
        in_planes = nChannels[2] if i == 0 else nChannels[3]
        x = basic_block(x, in_planes, nChannels[3], stride, stage=3, block=i)
        
    # Final BN + ReLU + AvgPool + FC
    x = layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="bn1")(x)
    x = layers.Activation('relu', name="relu_final")(x)
    x = layers.GlobalAveragePooling2D(name="avgpool")(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="fc")(x)
    
    return models.Model(inputs, outputs, name="WRN-40-2_Fused")

def convert_wrn_weights_fixed(pth_path, output_h5):
    print(f"--- Loading Teacher WRN weights from {pth_path} ---")
    try:
        checkpoint = torch.load(pth_path, map_location='cpu', weights_only=False)
    except FileNotFoundError:
        print(f"Error: {pth_path} not found.")
        return

    # Unwrap state dict
    if 'model' in checkpoint: state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
    else: state_dict = checkpoint
    
    # Normalize keys
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    print("Building Keras WRN-40-2 (Fused Pre-Act)...")
    model = build_wrn_40_2_fused()
    
    # CIFAR-100 Stats
    mean = np.array([0.5071, 0.4867, 0.4408], dtype=np.float32)
    std = np.array([0.2675, 0.2565, 0.2761], dtype=np.float32)
    
    def get_w(keras_name, param_type):
        # 1. Global Layers
        if keras_name == "conv1": return "conv1"
        if keras_name == "bn1": return "bn1"
        if keras_name == "fc": return "fc"
        
        # 2. Block Layers (layerX.Y.something)
        parts = keras_name.split('.')
        if parts[0].startswith("layer"):
            stage = parts[0].replace("layer", "") 
            block = parts[1] 
            layer_type = ".".join(parts[2:]) 
            
            # Map "downsample.0" -> "convShortcut"
            if "downsample.0" in layer_type:
                layer_type = layer_type.replace("downsample.0", "convShortcut")
            
            return f"block{stage}.layer.{block}.{layer_type}"
            
        raise ValueError(f"Unknown Keras layer name: {keras_name}")

    def load_param(key):
        if key not in state_dict:
            raise KeyError(f"{key} not found")
        return state_dict[key].numpy()

    count = 0
    print("--- Mapping & Fusing Weights ---")
    
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D):
            # 1. HANDLE FUSION (First Layer)
            if layer.name == "conv1":
                print(f"  > Fusing Normalization into {layer.name}...")
                w_torch = load_param("conv1.weight")
                
                std_broad = std.reshape(1, 3, 1, 1)
                w_fused = w_torch / std_broad
                w_sum_spatial = np.sum(w_fused, axis=(2, 3))
                bias_fused = np.sum(w_sum_spatial * (-mean), axis=1)
                
                w_keras = np.transpose(w_fused, (2, 3, 1, 0))
                layer.set_weights([w_keras, bias_fused])
                count += 1
                continue

            # 2. STANDARD CONV
            base_key = get_w(layer.name, "weight")
            w_torch = load_param(f"{base_key}.weight")
            w_keras = np.transpose(w_torch, (2, 3, 1, 0))
            
            if layer.use_bias:
                b_torch = load_param(f"{base_key}.bias")
                layer.set_weights([w_keras, b_torch])
            else:
                layer.set_weights([w_keras])
            count += 1
            
        elif isinstance(layer, layers.BatchNormalization):
            base_key = get_w(layer.name, "weight")
            gamma = load_param(f"{base_key}.weight")
            beta  = load_param(f"{base_key}.bias")
            mean_v = load_param(f"{base_key}.running_mean")
            var_v  = load_param(f"{base_key}.running_var")
            layer.set_weights([gamma, beta, mean_v, var_v])
            count += 1
            
        elif isinstance(layer, layers.Dense):
            w_torch = load_param("fc.weight")
            b_torch = load_param("fc.bias")
            layer.set_weights([w_torch.T, b_torch])
            count += 1
            
    print(f"--- Success! Transferred {count} layers. ---")
    model.save(output_h5)
    print(f"--- Saved to {output_h5} ---")

if __name__ == "__main__":
    convert_wrn_weights_fixed("wrn_40_2_75.61.pth", "wrn_40_2_teacher.h5")