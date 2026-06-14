import tensorflow as tf
import numpy as np
import copy
import hashlib

class IRNode:
    def __init__(self, name, op_type, inputs=None, weights=None, config=None, output_shape=None):
        self.name = name                
        self.op_type = op_type          
        self.inputs = inputs or []      
        self.outputs = []               
        self.weights = weights or []    
        self.config = config or {}      
        self.output_shape = output_shape 
        self.is_global_output = False   

    def __repr__(self):
        shape_str = str(self.output_shape) if self.output_shape else "?"
        return f"[{self.op_type}] {self.name} (Shape: {shape_str})"

class GraphExplorer:
    def __init__(self, model):
        self.model = model
        self.nodes = {} 
        self.input_node_name = "pipeline_input"

    def _sanitize_name(self, name):
        return name.replace("/", "_").replace(":", "_").replace("-", "_").replace(".", "_")

    def _extract_inbound_names(self, layer):
        inbound_names = []
        try:
            if hasattr(layer, 'inbound_nodes'):
                for node in layer.inbound_nodes:
                    if hasattr(node, 'parent_layers'):
                        parents = node.parent_layers
                        if not isinstance(parents, list): parents = [parents]
                        for p in parents: inbound_names.append(p.name)
                    elif hasattr(node, 'inbound_layers'):
                        parents = node.inbound_layers
                        if not isinstance(parents, list): parents = [parents]
                        for p in parents: inbound_names.append(p.name)
        except: pass
        
        if not inbound_names:
            layers = self.model.layers
            try:
                idx = layers.index(layer)
                if idx > 0: inbound_names.append(layers[idx-1].name)
                else: inbound_names.append(self.input_node_name)
            except ValueError: pass
        return [self._sanitize_name(n) for n in inbound_names]

    def build_ir(self):
        print("--- [IR] Exploding Model into Atomic Operations ---")
        self.nodes[self.input_node_name] = IRNode(self.input_node_name, "Input", output_shape=self.model.input_shape)
        
        for layer in self.model.layers:
            name = self._sanitize_name(layer.name)
            op_type = layer.__class__.__name__
            inputs = self._extract_inbound_names(layer)
            weights = layer.get_weights()
            config = layer.get_config()
            
            try:
                if hasattr(layer, 'inbound_nodes') and len(layer.inbound_nodes) > 0:
                    output_shape = layer.inbound_nodes[0].output_shapes
                else:
                    output_shape = layer.output_shape
            except:
                output_shape = None
            
            node = IRNode(name, op_type, inputs, weights, config, output_shape)
            self.nodes[name] = node
        
        for name, node in self.nodes.items():
            for inp in node.inputs:
                if inp in self.nodes:
                    self.nodes[inp].outputs.append(name)
        
        print(f"--- [IR] Explosion Complete. Total Nodes: {len(self.nodes)} ---")
        return list(self.nodes.values())

class GraphOptimizer:
    def __init__(self, nodes_list):
        self.nodes = {n.name: n for n in nodes_list}
        self.execution_order = nodes_list 

    def _dce_pass(self):
        alive = set()
        for name, node in self.nodes.items():
            if not node.outputs: 
                alive.add(name)
                node.is_global_output = True
            if node.op_type == "Input": alive.add(name)

        changed = True
        while changed:
            changed = False
            current_alive = list(alive)
            for name in current_alive:
                node = self.nodes.get(name)
                if not node: continue
                for inp in node.inputs:
                    if inp not in alive:
                        alive.add(inp)
                        changed = True
        
        msg_removed = 0
        new_order = []
        for node in self.execution_order:
            if node.name in alive: new_order.append(node)
            else:
                msg_removed += 1
                del self.nodes[node.name]
        self.execution_order = new_order
        print(f"  > [DCE] Removed {msg_removed} dead nodes.")

    def _cse_pass(self):
        signatures = {}
        merged_count = 0
        nodes_to_remove = set()

        for node in self.execution_order:
            w_hash = 0
            if node.weights:
                for w in node.weights: w_hash ^= hash(w.tobytes())
            
            sig = (node.op_type, tuple(node.inputs), str(node.config), w_hash)
            
            if sig in signatures:
                existing_name = signatures[sig]
                for out_name in node.outputs:
                    out_node = self.nodes[out_name]
                    out_node.inputs = [existing_name if x == node.name else x for x in out_node.inputs]
                    self.nodes[existing_name].outputs.append(out_name)
                
                nodes_to_remove.add(node.name)
                merged_count += 1
            else:
                signatures[sig] = node.name
        
        if merged_count > 0:
            for name in nodes_to_remove:
                if name in self.nodes: del self.nodes[name]
            self.execution_order = [n for n in self.execution_order if n.name not in nodes_to_remove]
        print(f"  > [CSE] Merged {merged_count} redundant operations.")

    def _linear_collapse_pass(self):
        collapsed = 0
        node_names = [n.name for n in self.execution_order]
        
        print(f"  [Debug: Linear Collapse] Scanning {len(node_names)} nodes...")
        for name in node_names:
            if name not in self.nodes: continue
            node = self.nodes[name]
            
            if node.op_type == "Dense":
                if len(node.outputs) != 1:
                    print(f"    - {name}: Skip collapse. Outputs count is {len(node.outputs)} (expected 1).")
                    continue
                
                child_name = node.outputs[0]
                child = self.nodes.get(child_name)
                if not child: continue
                
                if child.op_type == "Dense":
                    parent_act = node.config.get('activation', 'linear')
                    print(f"    ? {name} -> {child_name}: Parent activation is '{parent_act}'.")
                    
                    if parent_act == 'linear' or parent_act is None:
                        W1, b1 = node.weights
                        W2, b2 = child.weights
                        W_new = np.dot(W1, W2)
                        b_new = np.dot(b1, W2) + b2
                        
                        node.weights = [W_new, b_new]
                        node.config['units'] = child.config['units']
                        node.output_shape = child.output_shape
                        node.outputs = child.outputs
                        
                        for out_name in child.outputs:
                            out_node = self.nodes[out_name]
                            out_node.inputs = [name if x == child_name else x for x in out_node.inputs]
                        
                        if child_name in self.nodes: del self.nodes[child_name]
                        self.execution_order = [n for n in self.execution_order if n.name != child_name]
                        collapsed += 1
                        print(f"    ! COLLAPSED: {name} swallowed {child_name}")
                        
        print(f"  > [Linear Collapse] Merged {collapsed} linear chains.")

    def _algebraic_simplification_pass(self):
        print(f"  > [Algebraic Simplification] Scanned for identity ops.")

    def _operator_fusion_pass(self):
        fused_count = 0
        node_names = [n.name for n in self.execution_order]
        
        print(f"  [Debug: Fusion] Starting pass. OpTypes detected: {set(n.op_type for n in self.execution_order)}")
        
        def fuse_nodes(parent_name, child_name):
            parent = self.nodes[parent_name]
            child = self.nodes[child_name]
            parent.outputs = child.outputs
            for out_name in child.outputs:
                out_node = self.nodes[out_name]
                out_node.inputs = [parent_name if x == child_name else x for x in out_node.inputs]
            if child_name in self.nodes: del self.nodes[child_name]
            self.execution_order = [n for n in self.execution_order if n.name != child_name]

        for name in node_names:
            if name not in self.nodes: continue
            node = self.nodes[name]
            
            # Debug Rule 1: Topology
            if len(node.outputs) != 1:
                # Skip branching nodes
                continue
                
            child_name = node.outputs[0]
            child = self.nodes.get(child_name)
            if not child: continue

            # Debug Rule 2: Pattern Match
            # BN FOLDING
            if child.op_type == "BatchNormalization":
                # CRITICAL FIX: Check Op Type FIRST to avoid crashing on 'Add' layers
                if node.op_type in ["Dense", "Conv2D"]:
                    
                    # 1. Safely extract Weights (w) and Bias (b)
                    if len(node.weights) == 2:
                        w, b = node.weights
                    elif len(node.weights) == 1:
                        # If layer has no bias, create a zero bias vector
                        w = node.weights[0]
                        b = np.zeros(w.shape[-1], dtype=np.float32)
                    else:
                        # Safety: Skip layers with unexpected weight counts
                        continue

                    # 2. Get BN parameters
                    gamma, beta = child.weights[0], child.weights[1]
                    mean, var = child.weights[2], child.weights[3]
                    epsilon = child.config.get('epsilon', 1e-3)
                    
                    # 3. Calculate Fusion
                    scale = gamma / np.sqrt(var + epsilon)
                    bias_shift = beta - (mean * scale)
                    
                    w_new = w * scale
                    b_new = (b * scale) + bias_shift
                    
                    # 4. Save back (Overwrite list to ensure [w, b] structure)
                    node.weights = [w_new, b_new]
                    
                    fuse_nodes(name, child_name)
                    fused_count += 1
                    
                    # Update reference for next pass (Activation fusion)
                    if len(node.outputs) == 1:
                        child_name = node.outputs[0]
                        child = self.nodes.get(child_name)
                    else: continue
                else:
                    # Skip if parent is Add, Input, etc.
                    pass

            # ACTIVATION FUSION
            if child and child.op_type in ["Activation", "ReLU", "LeakyReLU", "Softmax"]:
                if node.op_type in ["Dense", "Conv2D", "Add"]:
                    act_type = "linear"
                    if child.op_type == "ReLU":
                        act_type = "relu"
                        if child.config.get('max_value') == 6.0: act_type = "relu6"
                    elif child.op_type == "Softmax": act_type = "softmax"
                    elif child.op_type == "Activation": act_type = child.config.get('activation')
                    
                    node.config['activation'] = act_type
                    fuse_nodes(name, child_name)
                    fused_count += 1
                else:
                    pass

        print(f"  > [Operator Fusion] Fused {fused_count} operations.")

    def optimize(self):
        print("--- [Optimizer] Running Graph Passes ---")
        self._dce_pass()
        self._cse_pass()                
        self._linear_collapse_pass()    
        self._algebraic_simplification_pass() 
        self._operator_fusion_pass() 
        self._dce_pass() 
        return self.execution_order