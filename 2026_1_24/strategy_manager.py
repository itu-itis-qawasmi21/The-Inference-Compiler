# strategy_manager.py
from pruning import PruningEngine
from quantization import QuantizationEngine
from graph_optimizer import GraphExplorer, GraphOptimizer
from compiler import Compiler

class OptimizationStrategy:
    """
    Orchestrates the full optimization pipeline:
    1. Pruning (Sparsity)
    2. Quantization (Precision Reduction)
    3. Graph Analysis (IR Generation)
    4. Graph Optimization (Fusion, DCE)
    5. Code Compilation (C++ Generation)
    """
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def execute_pipeline(self, train_ds, val_x, val_y, sparsity_target, quant_threshold, output_name="model_distilled.cpp"):
        print("\n=== Starting Optimization Pipeline ===")
        
        # ---------------------------------------------------------
        # 1. PRUNING
        # ---------------------------------------------------------
        print("--- Phase 1: Pruning ---")
        # Uses the complex PruningEngine (Greedy -> Heuristic -> Weights)
        pruner = PruningEngine(self.model, self.config)
        pruned_model = pruner.run_iterative_pruning(
            train_data=train_ds, 
            val_data=(val_x, val_y), 
            target_sparsity=sparsity_target
        )
        
        # ---------------------------------------------------------
        # 2. QUANTIZATION
        # ---------------------------------------------------------
        print("--- Phase 2: Quantization ---")
        # Uses 'power_of_2' mode for hardware efficiency + 32-bit Bias fix
        quant_mode = 'power_of_2' 
        quantizer = QuantizationEngine(pruned_model)
        
        # Per-layer sensitivity analysis
        sensitivity = quantizer.analyze_sensitivity(val_x, val_y, mode=quant_mode)
        
        # Application with Identity Snapping
        qat_model = quantizer.apply_qat(sensitivity, accuracy_threshold=quant_threshold, mode=quant_mode)
        
        # ---------------------------------------------------------
        # 3. INTERMEDIATE REPRESENTATION (IR)
        # ---------------------------------------------------------
        print("--- Phase 3: Intermediate Representation (IR) ---")
        # Explodes Keras model into generic IRNodes
        explorer = GraphExplorer(qat_model)
        ir_nodes = explorer.build_ir()

        # ---------------------------------------------------------
        # 4. GRAPH OPTIMIZATION
        # ---------------------------------------------------------
        print("--- Phase 4: Graph Optimization ---")
        # Runs DCE, CSE, BN-Folding, and Add+ReLU Fusion
        optimizer = GraphOptimizer(ir_nodes)
        optimized_graph_list = optimizer.optimize()

        # ---------------------------------------------------------
        # 5. COMPILATION
        # ---------------------------------------------------------
        print(f"--- Phase 5: Code Generation ({output_name}) ---")
        # Generates Branchless C++ with OpenMP Parallelism
        compiler = Compiler(optimized_graph_list)
        compiler.compile(output_name)

        # Return the finalized Keras model for Python-side evaluation
        # and the graph list if debugging is needed
        return optimized_graph_list, qat_model