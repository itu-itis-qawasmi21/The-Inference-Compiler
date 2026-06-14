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

    def execute_pipeline(
        self,
        train_ds,
        val_x,
        val_y,
        sparsity_target,
        quant_threshold,
        output_name="model_distilled.cpp",
        pruning_mode="recovery",   # 'recovery' | 'standard'
    ):
        print("\n=== Starting Optimization Pipeline ===")
        print(f"    Pruning Mode : {pruning_mode}")

        # ---------------------------------------------------------
        # 1. PRUNING
        # ---------------------------------------------------------
        print("--- Phase 1: Pruning ---")
        pruner = PruningEngine(self.model, self.config, pruning_mode=pruning_mode)
        pruned_model = pruner.run_iterative_pruning(
            train_data=train_ds,
            val_data=(val_x, val_y),
            target_sparsity=sparsity_target
        )

        # ---------------------------------------------------------
        # 2. QUANTIZATION
        # ---------------------------------------------------------
        print("--- Phase 2: Quantization ---")
        quant_mode = 'power_of_2'
        quantizer = QuantizationEngine(pruned_model)
        sensitivity = quantizer.analyze_sensitivity(val_x, val_y, mode=quant_mode)
        qat_model = quantizer.apply_qat(
            sensitivity, accuracy_threshold=quant_threshold, mode=quant_mode
        )

        # ---------------------------------------------------------
        # 3. INTERMEDIATE REPRESENTATION (IR)
        # ---------------------------------------------------------
        print("--- Phase 3: Intermediate Representation (IR) ---")
        explorer = GraphExplorer(qat_model)
        ir_nodes = explorer.build_ir()

        # ---------------------------------------------------------
        # 4. GRAPH OPTIMIZATION
        # ---------------------------------------------------------
        print("--- Phase 4: Graph Optimization ---")
        optimizer = GraphOptimizer(ir_nodes)
        optimized_graph_list = optimizer.optimize()

        # ---------------------------------------------------------
        # 5. COMPILATION
        # ---------------------------------------------------------
        print(f"--- Phase 5: Code Generation ({output_name}) ---")
        compiler = Compiler(optimized_graph_list)
        compiler.compile(output_name)

        return optimized_graph_list, qat_model
