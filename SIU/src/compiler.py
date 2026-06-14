import numpy as np
import os

class Compiler:
    """
    High-Performance C++ Compiler Backend.
    """
    def __init__(self, graph_nodes):
        self.graph = graph_nodes 
        self.buffer_map = {}     
        self.code_lines = []     

    def _sanitize(self, name):
        return name.replace(".", "_").replace("/", "_").replace(":", "_").replace("-", "_")

    def _get_buffer_name(self, node_name):
        if node_name in self.buffer_map: return self.buffer_map[node_name]
        clean_name = self._sanitize(node_name)
        if clean_name in self.buffer_map: return self.buffer_map[clean_name]
        if "input" in clean_name.lower() or "pipeline" in clean_name.lower(): return "input"
        raise KeyError(f"Compiler Error: Buffer for parent '{node_name}' not found.")

    def _get_activation_code(self, val_name, activation_type):
        if activation_type in ['relu', 'relu6']: 
            return f"std::max(0.0f, {val_name})"
        elif activation_type == 'sigmoid': 
            return f"1.0f / (1.0f + exp(-{val_name}))"
        elif activation_type == 'tanh': 
            return f"tanh({val_name})"
        return val_name 

    def _get_volume(self, node):
        """FIX 1: Safely handles both tuples and lists of tuples for shape extraction"""
        if not hasattr(node, 'output_shape') or not node.output_shape:
            return 32768 # Fallback safe volume
        
        shape = node.output_shape
        if isinstance(shape, list) and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
            
        dims = [d for d in shape if d is not None]
        if not dims: return 32768
        return int(np.prod(dims))

    def _generate_identity(self, node):
        input_name = self._get_buffer_name(node.inputs[0])
        self.buffer_map[node.name] = input_name 
        self.code_lines.append(f"    // [{node.op_type}] {node.name} -> Alias to {input_name}")

    def _generate_add(self, node):
        if len(node.inputs) < 2:
            self._generate_identity(node)
            return

        buf0 = self._get_buffer_name(node.inputs[0])
        buf1 = self._get_buffer_name(node.inputs[1])
        output_name = f"buff_{self._sanitize(node.name)}"
        self.buffer_map[node.name] = output_name
        
        size = self._get_volume(node)
        activation = node.config.get('activation')

        self.code_lines.append(f"    // [Add] {node.name} (Fused: {activation})")
        self.code_lines.append(f"    static float {output_name}[{size}];")
        self.code_lines.append(f"    #pragma omp parallel for")
        self.code_lines.append(f"    for(int i=0; i<{size}; i++) {{")
        self.code_lines.append(f"        float sum = {buf0}[i] + {buf1}[i];")
        
        if activation and activation != 'linear':
            act_code = self._get_activation_code("sum", activation)
            self.code_lines.append(f"        {output_name}[i] = {act_code};")
        else:
            self.code_lines.append(f"        {output_name}[i] = sum;")
        self.code_lines.append(f"    }}")
        self.code_lines.append("")

    def _generate_scale_bias(self, node):
        scale, shift = node.weights
        input_name  = self._get_buffer_name(node.inputs[0])
        output_name = f"buff_{self._sanitize(node.name)}"
        self.buffer_map[node.name] = output_name

        size = self._get_volume(node)
        channels = len(scale)

        scale_str = ", ".join([f"{v:.8f}f" for v in scale])
        shift_str = ", ".join([f"{v:.8f}f" for v in shift])

        self.code_lines.append(f"    // [ScaleBias] {node.name} (folded BN)")
        self.code_lines.append(f"    static float {output_name}[{size}];")
        self.code_lines.append(f"    static const float scale_{output_name}[] = {{{scale_str}}};")
        self.code_lines.append(f"    static const float shift_{output_name}[] = {{{shift_str}}};")
        self.code_lines.append(f"    #pragma omp parallel for")
        self.code_lines.append(f"    for(int i=0; i<{size}; i++) {{")
        self.code_lines.append(f"        int ch = i % {channels};")
        self.code_lines.append(f"        {output_name}[i] = scale_{output_name}[ch] * {input_name}[i] + shift_{output_name}[ch];")
        self.code_lines.append(f"    }}")
        self.code_lines.append("")

    def _generate_standalone_activation(self, node):
        input_name  = self._get_buffer_name(node.inputs[0])
        output_name = f"buff_{self._sanitize(node.name)}"
        self.buffer_map[node.name] = output_name
        activation  = node.config.get('activation', 'linear')

        size = self._get_volume(node)

        self.code_lines.append(f"    // [StandaloneActivation] {node.name} ({activation})")
        self.code_lines.append(f"    static float {output_name}[{size}];")

        if activation == 'softmax':
            self.code_lines.append(f"    {{")
            self.code_lines.append(f"        float _max = {input_name}[0];")
            self.code_lines.append(f"        for(int i=1; i<{size}; i++) if({input_name}[i] > _max) _max = {input_name}[i];")
            self.code_lines.append(f"        float _sum = 0.0f;")
            self.code_lines.append(f"        for(int i=0; i<{size}; i++) {{ {output_name}[i] = exp({input_name}[i] - _max); _sum += {output_name}[i]; }}")
            self.code_lines.append(f"        for(int i=0; i<{size}; i++) {output_name}[i] /= _sum;")
            self.code_lines.append(f"    }}")
        else:
            act_code = self._get_activation_code(f"{input_name}[i]", activation)
            self.code_lines.append(f"    #pragma omp parallel for")
            self.code_lines.append(f"    for(int i=0; i<{size}; i++) {{")
            self.code_lines.append(f"        {output_name}[i] = {act_code};")
            self.code_lines.append(f"    }}")
        self.code_lines.append("")

    def _generate_dense_sparse(self, node):
        weights, bias = node.weights
        input_name = self._get_buffer_name(node.inputs[0])
        output_name = f"buff_{self._sanitize(node.name)}"
        self.buffer_map[node.name] = output_name
        output_dim = weights.shape[1]
        activation = node.config.get('activation')

        self.code_lines.append(f"    // [Dense] {node.name}")
        self.code_lines.append(f"    static float {output_name}[{output_dim}];")
        
        for i in range(output_dim):
            bias_val = bias[i]
            self.code_lines.append(f"    {{")
            self.code_lines.append(f"        float acc = {bias_val:.8f}f;")
            col_weights = weights[:, i]
            non_zeros = np.where(np.abs(col_weights) > 1e-9)[0]
            
            for idx in non_zeros:
                w_val = col_weights[idx]
                self.code_lines.append(f"        acc += {w_val:.8f}f * {input_name}[{idx}];")
            
            if activation == 'softmax':
                self.code_lines.append(f"        {output_name}[{i}] = acc;")
            else:
                act_code = self._get_activation_code("acc", activation)
                self.code_lines.append(f"        {output_name}[{i}] = {act_code};")
            self.code_lines.append(f"    }}")

        if activation == 'softmax':
            self.code_lines.append(f"    {{")
            self.code_lines.append(f"        float _max = {output_name}[0];")
            self.code_lines.append(f"        for(int i=1; i<{output_dim}; i++) if({output_name}[i] > _max) _max = {output_name}[i];")
            self.code_lines.append(f"        float _sum = 0.0f;")
            self.code_lines.append(f"        for(int i=0; i<{output_dim}; i++) {{ {output_name}[i] = exp({output_name}[i] - _max); _sum += {output_name}[i]; }}")
            self.code_lines.append(f"        for(int i=0; i<{output_dim}; i++) {output_name}[i] /= _sum;")
            self.code_lines.append(f"    }}")
        self.code_lines.append("")

    def _generate_conv2d(self, node):
        if len(node.weights) == 2:
            weights, bias = node.weights
        else:
            weights = node.weights[0]
            bias = np.zeros(weights.shape[-1], dtype=np.float32)

        input_name = self._get_buffer_name(node.inputs[0])
        output_name = f"buff_{self._sanitize(node.name)}"
        self.buffer_map[node.name] = output_name
        
        kh, kw, cin, cout = weights.shape
        stride = node.config.get('strides', (1, 1))[0]
        padding = node.config.get('padding', 'valid').upper()
        activation = node.config.get('activation')
        
        # Calculate Dimensions and Padding
        shape = node.output_shape
        if isinstance(shape, list) and isinstance(shape[0], (tuple, list)): shape = shape[0]
        dims = [d for d in shape if d is not None] if shape else []
        h_out, w_out = (dims[0], dims[1]) if len(dims) >= 2 else (32, 32)

        if padding == 'SAME':
            h_in = h_out * stride
            w_in = w_out * stride
            pad_offset = (kh - 1) // 2
        else:
            h_in = (h_out - 1) * stride + kh
            w_in = (w_out - 1) * stride + kw
            pad_offset = 0

        self.code_lines.append(f"    // [Conv2D] {node.name}")
        self.code_lines.append(f"    static float {output_name}[{h_out * w_out * cout}];")
        
        # FIX 2: Flatten all weights into ONE array indexed as (cout, kh, kw, cin)
        w_flat = weights.transpose(3, 0, 1, 2).flatten()
        w_str = ", ".join([f"{v:.8f}f" for v in w_flat])
        self.code_lines.append(f"    static const float w_{output_name}[] = {{{w_str}}};")
        
        b_str = ", ".join([f"{v:.8f}f" for v in bias])
        self.code_lines.append(f"    static const float b_{output_name}[] = {{{b_str}}};")
        
        # Convolution loop with FIX 3: Safe padding boundaries
        self.code_lines.append(f"    #pragma omp parallel for collapse(2)")
        self.code_lines.append(f"    for(int oh=0; oh<{h_out}; oh++) {{")
        self.code_lines.append(f"        for(int ow=0; ow<{w_out}; ow++) {{")
        self.code_lines.append(f"            for(int co=0; co<{cout}; co++) {{")
        self.code_lines.append(f"                float sum = b_{output_name}[co];")
        self.code_lines.append(f"                for(int kh=0; kh<{kh}; kh++) {{")
        self.code_lines.append(f"                    for(int kw=0; kw<{kw}; kw++) {{")
        self.code_lines.append(f"                        for(int ci=0; ci<{cin}; ci++) {{")
        self.code_lines.append(f"                            int ih = oh * {stride} + kh - {pad_offset};")
        self.code_lines.append(f"                            int iw = ow * {stride} + kw - {pad_offset};")
        self.code_lines.append(f"                            if(ih >= 0 && ih < {h_in} && iw >= 0 && iw < {w_in}) {{")
        self.code_lines.append(f"                                int in_idx = (ih * {w_in} + iw) * {cin} + ci;")
        self.code_lines.append(f"                                int w_idx = (co * {kh * kw * cin}) + (kh * {kw * cin}) + (kw * {cin}) + ci;")
        self.code_lines.append(f"                                sum += {input_name}[in_idx] * w_{output_name}[w_idx];")
        self.code_lines.append(f"                            }}") # Close If
        self.code_lines.append(f"                        }}")   # close ci
        self.code_lines.append(f"                    }}")       # close kw
        self.code_lines.append(f"                }}")           # close kh
        
        if activation and activation != 'linear':
            act_code = self._get_activation_code("sum", activation)
            self.code_lines.append(f"                {output_name}[(oh * {w_out} + ow) * {cout} + co] = {act_code};")
        else:
            self.code_lines.append(f"                {output_name}[(oh * {w_out} + ow) * {cout} + co] = sum;")
        
        self.code_lines.append(f"            }}")   # close co
        self.code_lines.append(f"        }}")       # close ow
        self.code_lines.append(f"    }}")           # close oh
        self.code_lines.append("")

    def _generate_global_avg_pool(self, node):
        input_name = self._get_buffer_name(node.inputs[0])
        output_name = f"buff_{self._sanitize(node.name)}"
        self.buffer_map[node.name] = output_name
        
        shape = node.inputs[0].output_shape if hasattr(node.inputs[0], 'output_shape') else None
        if isinstance(shape, list) and isinstance(shape[0], (tuple, list)): shape = shape[0]
        dims = [d for d in shape if d is not None] if shape else []
        
        spatial_size = dims[0] * dims[1] if len(dims) >= 3 else 1024
        channels = dims[2] if len(dims) >= 3 else 64
        
        self.code_lines.append(f"    // [GlobalAvgPool] {node.name}")
        self.code_lines.append(f"    static float {output_name}[{channels}];")
        self.code_lines.append(f"    #pragma omp parallel for")
        self.code_lines.append(f"    for(int c=0; c<{channels}; c++) {{")
        self.code_lines.append(f"       float sum = 0.0f;")
        self.code_lines.append(f"       for(int i=0; i<{spatial_size}; i++) sum += {input_name}[i * {channels} + c];")
        self.code_lines.append(f"       {output_name}[c] = sum / {spatial_size}.0f; }}")
        self.code_lines.append("")

    def compile(self, filename="model_distilled.cpp"):
        print(f"--- [Compiler] Generating C++ Code to {filename} ---")
        self.code_lines = []
        self.code_lines.append("#include <cmath>")
        self.code_lines.append("#include <algorithm>")
        self.code_lines.append("#include <iostream>")
        self.code_lines.append("using namespace std;\n")
        self.code_lines.append("void run_inference(const float* input, float* output) {")
        
        try:
            input_node = next(n for n in self.graph if n.op_type == "Input")
            self.buffer_map[input_node.name] = "input"
        except StopIteration:
            self.buffer_map["pipeline_input"] = "input"
        
        for node in self.graph:
            if node.op_type == "Input": continue
            if node.op_type == "Dense": self._generate_dense_sparse(node)
            elif node.op_type == "Conv2D": self._generate_conv2d(node)
            elif node.op_type == "GlobalAveragePooling2D": self._generate_global_avg_pool(node)
            elif node.op_type == "Add": self._generate_add(node)
            elif node.op_type == "ScaleBias": self._generate_scale_bias(node)
            elif node.op_type == "StandaloneActivation": self._generate_standalone_activation(node)
            elif node.op_type in ["Flatten", "Dropout", "Reshape", "InputLayer"]: self._generate_identity(node)
            else:
                print(f"  [Compiler Warning] Unknown Op: {node.op_type} -> Treated as Identity")
                if node.inputs: self._generate_identity(node)
        
        last_node = self.graph[-1]
        last_buffer = self._get_buffer_name(last_node.name)
        out_size = 10 
        
        shape = last_node.output_shape if hasattr(last_node, 'output_shape') else None
        if isinstance(shape, list) and isinstance(shape[0], (tuple, list)): shape = shape[0]
        if shape and shape[-1] is not None: out_size = shape[-1]
             
        self.code_lines.append(f"    // Copy Result")
        self.code_lines.append(f"    for(int i=0; i<{out_size}; i++) output[i] = {last_buffer}[i];")
        self.code_lines.append("}")
        
        with open(filename, "w") as f: f.write("\n".join(self.code_lines))
        print(f"  > Code Generation Complete. ({len(self.code_lines)} lines)")