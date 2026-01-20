import numpy as np
import os

class Compiler:
    """
    High-Performance C++ Compiler Backend.
    
    IMPLEMENTS:
    - Sparse Algebraic Unrolling: Weights are hardcoded as literals (x * 0.054).
    - Loop Unrolling: Dense layers have NO loops for weights, only unrolled sums.
    - Branchless Logic: Activations use std::max / bitwise ops.
    - ResNet Support: Compiles 'Add' layers with fused activations.
    - OpenMP Parallelism: Auto-vectorizes loops across CPU cores.
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
        """Branchless / Optimized Activation Implementations"""
        if activation_type in ['relu', 'relu6']: 
            # Branchless max for ReLU
            return f"std::max(0.0f, {val_name})"
        elif activation_type == 'sigmoid': 
            return f"1.0f / (1.0f + exp(-{val_name}))"
        elif activation_type == 'tanh': 
            return f"tanh({val_name})"
        return val_name 

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
        
        size = 0
        if hasattr(node, 'output_shape') and node.output_shape:
            size = int(np.prod(node.output_shape[1:]))
        
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

    def _generate_dense_sparse(self, node):
        """
        ALGEBRAIC UNROLLING:
        Instead of loops, we write: acc += 0.54f * input[2];
        This is constant folding + loop unrolling + sparse optimization all in one.
        """
        weights, bias = node.weights
        input_name = self._get_buffer_name(node.inputs[0])
        output_name = f"buff_{self._sanitize(node.name)}"
        self.buffer_map[node.name] = output_name
        output_dim = weights.shape[1]
        activation = node.config.get('activation')

        self.code_lines.append(f"    // [Dense] {node.name} (Unrolled)")
        self.code_lines.append(f"    static float {output_name}[{output_dim}];")
        
        if activation == 'softmax':
            self.code_lines.append(f"    float {output_name}_sum = 0.0f; float {output_name}_max = -1e9f;") 

        if activation != 'softmax': self.code_lines.append(f"    #pragma omp parallel for")
        
        for i in range(output_dim):
            bias_val = bias[i]
            # HARDCODED CONSTANT (Bias)
            self.code_lines.append(f"    {{ float acc = {bias_val:.8f}f;")
            col_weights = weights[:, i]
            non_zeros = np.where(np.abs(col_weights) > 1e-9)[0]
            
            # UNROLLED LOOP (Weights)
            for idx in non_zeros:
                w_val = col_weights[idx]
                # HARDCODED CONSTANT (Weight)
                self.code_lines.append(f"      acc += {w_val:.8f}f * {input_name}[{idx}];")
            
            # BRANCHLESS ACTIVATION
            if activation != 'softmax':
                act_code = self._get_activation_code("acc", activation)
                self.code_lines.append(f"      {output_name}[{i}] = {act_code}; }}")
            else:
                self.code_lines.append(f"      {output_name}[{i}] = acc; }}")

        if activation == 'softmax':
            self.code_lines.append(f"    for(int i=0; i<{output_dim}; i++) if({output_name}[i] > {output_name}_max) {output_name}_max = {output_name}[i];")
            self.code_lines.append(f"    for(int i=0; i<{output_dim}; i++) {{ {output_name}[i] = exp({output_name}[i] - {output_name}_max); {output_name}_sum += {output_name}[i]; }}")
            self.code_lines.append(f"    for(int i=0; i<{output_dim}; i++) {output_name}[i] /= {output_name}_sum;")
        self.code_lines.append("") 

    def _generate_conv2d(self, node):
        # [RESNET FIX] Handle Conv2D layers with no bias
        if len(node.weights) == 2:
            weights, bias = node.weights
        else:
            weights = node.weights[0]
            # Create a zero-bias vector matching the number of output filters (last dim)
            bias = np.zeros(weights.shape[-1], dtype=np.float32)

        input_name = self._get_buffer_name(node.inputs[0])
        output_name = f"buff_{self._sanitize(node.name)}"
        self.buffer_map[node.name] = output_name
        kh, kw, cin, cout = weights.shape
        stride = node.config.get('strides', (1, 1))[0]
        padding = node.config.get('padding', 'valid').upper()
        activation = node.config.get('activation')
        
        if hasattr(node, 'output_shape') and node.output_shape: _, h_out, w_out, _ = node.output_shape
        else: h_out, w_out = 32, 32 

        self.code_lines.append(f"    // [Conv2D] {node.name}")
        self.code_lines.append(f"    static float {output_name}[{h_out * w_out * cout}];")
        
        flat_weights = weights.flatten()
        w_str = ",".join([f"{x:.6f}f" for x in flat_weights])
        self.code_lines.append(f"    static const float w_{output_name}[] = {{{w_str}}};")
        b_str = ",".join([f"{x:.6f}f" for x in bias])
        self.code_lines.append(f"    static const float b_{output_name}[] = {{{b_str}}};")

        self.code_lines.append(f"    #pragma omp parallel for collapse(2)")
        self.code_lines.append(f"    for(int h=0; h<{h_out}; h++) {{")
        self.code_lines.append(f"      for(int w=0; w<{w_out}; w++) {{")
        self.code_lines.append(f"        for(int co=0; co<{cout}; co++) {{")
        self.code_lines.append(f"          float sum = b_{output_name}[co];")
        self.code_lines.append(f"          for(int kh=0; kh<{kh}; kh++) {{")
        self.code_lines.append(f"            for(int kw=0; kw<{kw}; kw++) {{")
        self.code_lines.append(f"              for(int ci=0; ci<{cin}; ci++) {{")
        pad_offset = 1 if padding == 'SAME' else 0
        self.code_lines.append(f"                int h_in = h * {stride} + kh - {pad_offset};")
        self.code_lines.append(f"                int w_in = w * {stride} + kw - {pad_offset};")
        h_in_approx = h_out * stride 
        self.code_lines.append(f"                if(h_in >= 0 && h_in < {h_in_approx} && w_in >= 0 && w_in < {h_in_approx}) {{")
        self.code_lines.append(f"                  int in_idx = (h_in * {h_in_approx} + w_in) * {cin} + ci;")
        self.code_lines.append(f"                  int w_idx = (kh * {kw} * {cin} * {cout}) + (kw * {cin} * {cout}) + (ci * {cout}) + co;")
        self.code_lines.append(f"                  sum += {input_name}[in_idx] * w_{output_name}[w_idx];")
        self.code_lines.append(f"                }}")
        self.code_lines.append(f"              }} }} }}")
        act_code = self._get_activation_code("sum", activation)
        self.code_lines.append(f"          int out_idx = (h * {w_out} + w) * {cout} + co;")
        self.code_lines.append(f"          {output_name}[out_idx] = {act_code};")
        self.code_lines.append(f"        }} }} }}")
        self.code_lines.append("")

    def _generate_global_avg_pool(self, node):
        input_name = self._get_buffer_name(node.inputs[0])
        output_name = f"buff_{self._sanitize(node.name)}"
        self.buffer_map[node.name] = output_name
        channels = 64
        spatial_size = 64
        if hasattr(node, 'output_shape') and node.output_shape: channels = node.output_shape[-1]
        
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
        self.code_lines.append("#include <omp.h>")
        self.code_lines.append("using namespace std;")
        self.code_lines.append("")
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
            elif node.op_type in ["Flatten", "Dropout", "Reshape", "InputLayer"]: self._generate_identity(node)
            else:
                print(f"  [Compiler Warning] Unknown Op: {node.op_type} -> Treated as Identity")
                if node.inputs: self._generate_identity(node)
        
        last_node = self.graph[-1]
        last_buffer = self._get_buffer_name(last_node.name)
        out_size = 10 
        if hasattr(last_node, 'output_shape') and last_node.output_shape: out_size = last_node.output_shape[-1]
             
        self.code_lines.append(f"    // Copy Result")
        self.code_lines.append(f"    for(int i=0; i<{out_size}; i++) output[i] = {last_buffer}[i];")
        self.code_lines.append("}")
        
        with open(filename, "w") as f: f.write("\n".join(self.code_lines))
        print(f"  > Code Generation Complete. ({len(self.code_lines)} lines)")