import torch
import numpy as np
import tensorflow as tf
from keras import layers, models

def build_resnet110_cifar_fused(input_shape=(32, 32, 3), num_classes=100):
    """
    Defines ResNet-110 for CIFAR-100 with FUSED Input Normalization.
    The first Conv2D layer now has 'use_bias=True' to handle Mean subtraction.
    """
    # ResNet-110: n=18 blocks per stage.
    n = 18 
    
    def basic_block(x, filters, stride=1, stage=0, block=0):
        prefix = f"layer{stage}.{block}"
        identity = x
        
        # --- Conv 1 ---
        x = layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False, 
                          name=f"{prefix}.conv1")(x)
        x = layers.BatchNormalization(epsilon=1e-5, momentum=0.9, 
                                      name=f"{prefix}.bn1")(x)
        x = layers.Activation('relu', name=f"{prefix}.relu1")(x)
        
        # --- Conv 2 ---
        x = layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=False, 
                          name=f"{prefix}.conv2")(x)
        x = layers.BatchNormalization(epsilon=1e-5, momentum=0.9, 
                                      name=f"{prefix}.bn2")(x)
        
        # --- Downsample ---
        if stride != 1 or identity.shape[-1] != filters:
            identity = layers.Conv2D(filters, 1, strides=stride, padding='same', use_bias=False,
                                     name=f"{prefix}.downsample.0")(identity)
            identity = layers.BatchNormalization(epsilon=1e-5, momentum=0.9,
                                                 name=f"{prefix}.downsample.1")(identity)
            
        # --- Fusion Point ---
        x = layers.Add(name=f"{prefix}.add")([x, identity])
        x = layers.Activation('relu', name=f"{prefix}.relu2")(x)
        return x

    # --- Model Entry ---
    inputs = layers.Input(shape=input_shape)
    
    # [CRITICAL FIX] Enable Bias on First Conv to absorb Mean Subtraction
    x = layers.Conv2D(16, 3, padding='same', use_bias=True, name="conv1")(inputs)
    x = layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="bn1")(x)
    x = layers.Activation('relu', name="relu_init")(x)
    
    # --- Stack 1 (16 filters) ---
    for i in range(n):
        x = basic_block(x, 16, stride=1, stage=1, block=i)
        
    # --- Stack 2 (32 filters) ---
    x = basic_block(x, 32, stride=2, stage=2, block=0)
    for i in range(1, n):
        x = basic_block(x, 32, stride=1, stage=2, block=i)
        
    # --- Stack 3 (64 filters) ---
    x = basic_block(x, 64, stride=2, stage=3, block=0)
    for i in range(1, n):
        x = basic_block(x, 64, stride=1, stage=3, block=i)
        
    # --- Classifier ---
    x = layers.GlobalAveragePooling2D(name="avgpool")(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="fc")(x)
    
    return models.Model(inputs, outputs, name="ResNet110_Student_Fused")

def transfer_weights_fused(keras_model, torch_state_dict):
    """
    Maps PyTorch weights to Keras layers.
    Fuses CIFAR-100 Mean/Std normalization into the first Conv2D layer.
    """
    print(f"--- Starting Weight Transfer with Normalization Fusion ---")
    
    # CIFAR-100 Stats
    mean = np.array([0.5071, 0.4867, 0.4408], dtype=np.float32)
    std = np.array([0.2675, 0.2565, 0.2761], dtype=np.float32)

    def get_w(key):
        if key not in torch_state_dict:
            raise KeyError(f"PyTorch key '{key}' not found!")
        return torch_state_dict[key].numpy()

    count = 0
    for layer in keras_model.layers:
        if isinstance(layer, layers.Conv2D):
            # Special Handling for First Layer ("conv1")
            if layer.name == "conv1":
                print(f"  > Fusing Mean/Std into {layer.name}...")
                w_torch = get_w(f"{layer.name}.weight") # (Out, In, H, W)
                
                # 1. Fuse Std: w_new = w_old / std
                # Reshape Std for broadcasting: (1, In, 1, 1) -> (1, 3, 1, 1)
                std_broad = std.reshape(1, 3, 1, 1)
                w_fused = w_torch / std_broad
                
                # 2. Fuse Mean: bias_new = -Mean * Sum(w_fused over spatial)
                # Sum over H, W (last two dims in Torch) -> (Out, In)
                w_sum = np.sum(w_fused, axis=(2, 3))
                
                # Calculate bias shift: Sum_over_In( -Mean * w_sum )
                # Dot product: (In,) . (Out, In)^T -> (Out,)
                bias_fused = np.dot(w_sum, -mean)
                
                # 3. Transpose to Keras: (Out, In, H, W) -> (H, W, In, Out)
                w_keras = np.transpose(w_fused, (2, 3, 1, 0))
                
                layer.set_weights([w_keras, bias_fused])
                count += 1
                continue

            # Standard Conv2D Handling
            key = f"{layer.name}.weight"
            w_torch = get_w(key)
            w_keras = np.transpose(w_torch, (2, 3, 1, 0))
            
            if layer.use_bias:
                b_keras = get_w(f"{layer.name}.bias")
                layer.set_weights([w_keras, b_keras])
            else:
                layer.set_weights([w_keras])
            count += 1
            
        elif isinstance(layer, layers.BatchNormalization):
            gamma = get_w(f"{layer.name}.weight")
            beta  = get_w(f"{layer.name}.bias")
            mean_bn = get_w(f"{layer.name}.running_mean")
            var   = get_w(f"{layer.name}.running_var")
            layer.set_weights([gamma, beta, mean_bn, var])
            count += 1
            
        elif isinstance(layer, layers.Dense):
            w_torch = get_w(f"{layer.name}.weight")
            b_torch = get_w(f"{layer.name}.bias")
            w_keras = w_torch.T
            layer.set_weights([w_keras, b_torch])
            count += 1

    print(f"--- Successfully transferred weights for {count} layers ---")

def main():
    # Make sure this filename matches your actual uploaded file
    pth_path = "resnet110_74.31.pth" 
    output_h5 = "resnet110_optimized.h5"
    
    print(f"Loading {pth_path}...")
    try:
        checkpoint = torch.load(pth_path, map_location='cpu', weights_only=False)
    except FileNotFoundError:
        print(f"ERROR: {pth_path} not found. Please upload it!")
        return

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Check for 'module.' prefix (DataParallel) and remove it
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    print("Building Keras ResNet-110 (Fused)...")
    # Ensure num_classes matches your dataset (CIFAR-100)
    model = build_resnet110_cifar_fused(num_classes=100)

    transfer_weights_fused(model, state_dict)
    
    model.save(output_h5)
    print(f"Saved Fixed Keras model to: {output_h5}")

if __name__ == "__main__":
    main()