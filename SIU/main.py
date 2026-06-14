import tensorflow as tf
import numpy as np
import argparse
import os
import time
import sys
import datetime

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Path setup — src/ contains all importable modules
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

OUTPUTS_DIR = os.path.join(ROOT, "outputs")
MODELS_DIR  = os.path.join(ROOT, "models")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# --- CUSTOM MODULES (resolved via SRC on sys.path) ---
from knowledge_distillation import get_distiller
from strategy_manager import OptimizationStrategy
from evaluator import PerformanceEvaluator


# =========================================================
# 1. DATA LOADING
# =========================================================
def load_mnist_data():
    print("--- [Setup] Loading MNIST Data ---")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0
    x_train = x_train.reshape((-1, 784))
    x_test  = x_test.reshape((-1, 784))
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test  = tf.keras.utils.to_categorical(y_test,  10)
    return x_train, y_train, x_test, y_test


def load_cifar100_data():
    print("--- [Setup] Loading CIFAR-100 Data ---")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 100)
    y_test  = tf.keras.utils.to_categorical(y_test,  100)
    return x_train, y_train, x_test, y_test


# =========================================================
# 2. MODEL SETUP
# =========================================================
def load_teacher_model(model_path, input_shape, num_classes):
    print(f"--- [Setup] Loading Teacher Model from: {model_path} ---")
    try:
        model = tf.keras.models.load_model(model_path)
        print("  > Model loaded successfully.")
        return model
    except Exception as e:
        print(f"\n!!! FATAL ERROR LOADING MODEL: {e}")
        sys.exit(1)


def create_mlp_student(input_shape, num_classes):
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(784,)),
        tf.keras.layers.Dense(256, activation="relu", name="dense_0"),
        tf.keras.layers.Dense(128, activation="relu", name="dense_1"),
        tf.keras.layers.Dense(num_classes, activation="softmax", name="output"),
    ])


# =========================================================
# 3. MAIN PIPELINE
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",      type=str, default="mnist",
                        choices=["mnist", "cifar100"])
    parser.add_argument("--model_type",   type=str, default="mlp",
                        choices=["mlp", "resnet"])
    parser.add_argument("--model_path",   type=str, default=None)
    parser.add_argument("--distill",      type=str, default="none",
                        help="ours, external_hint, etc.")
    parser.add_argument("--prune",        action="store_true")
    parser.add_argument("--quantize",     action="store_true")
    parser.add_argument("--epochs",       type=int, default=1)
    parser.add_argument(
        "--pruning_mode",
        type=str,
        default="recovery",
        choices=["recovery", "standard"],
        help=(
            "'recovery' (default) uses Sensitivity-Aware Recovery in Phase 2. "
            "'standard' skips recovery logic — ablation baseline for results table."
        ),
    )
    args = parser.parse_args()

    # --- A. LOAD DATA ---
    if args.dataset == "mnist":
        x_train, y_train, x_test, y_test = load_mnist_data()
        input_shape = (784,)
        num_classes = 10
    elif args.dataset == "cifar100":
        x_train, y_train, x_test, y_test = load_cifar100_data()
        input_shape = (32, 32, 3)
        num_classes = 100

    # --- B. LOAD TEACHER ---
    if args.model_path is None:
        if args.model_type == "mlp":
            args.model_path = os.path.join(MODELS_DIR, "mnist_mlp_model.h5")
        else:
            print("!!! Error: --model_path is required for ResNet tests.")
            sys.exit(1)

    if not os.path.exists(args.model_path):
        print(f"!!! ERROR: Could not find model file: '{args.model_path}'")
        sys.exit(1)

    teacher_model = load_teacher_model(args.model_path, input_shape, num_classes)

    # --- C. PHASE 1: DISTILLATION ---
    if args.distill != "none":
        print(f"\n>>> Step 1: Distillation ({args.distill})")

        if args.model_type == "mlp":
            student_model = create_mlp_student(input_shape, num_classes)
        else:
            print("  > Cloning Teacher architecture as Student for Self-Distillation...")
            student_model = tf.keras.models.clone_model(teacher_model)
            student_model.set_weights(teacher_model.get_weights())

        start_time = time.time()
        n_samples = x_train.shape[0]
        distiller = get_distiller(args.distill, student_model, teacher_model, n_data=n_samples)

        optimizer        = tf.keras.optimizers.Adam(learning_rate=0.001)
        student_loss_fn  = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

        if args.distill == "ours":
            distiller.compile(
                optimizer=optimizer,
                metrics=["accuracy"],
                student_loss_fn=student_loss_fn,
                distillation_loss_fn=tf.keras.losses.KLDivergence(),
            )
        else:
            distiller.compile(
                optimizer=optimizer,
                metrics=["accuracy"],
                student_loss_fn=student_loss_fn,
            )

        distiller.fit(x_train, y_train, epochs=args.epochs, batch_size=64)
        print(f"  > Distillation Time: {time.time() - start_time:.2f}s")
        optimized_model = distiller.student
    else:
        print("\n>>> Skipping Distillation (Using Teacher as Base)")
        optimized_model = tf.keras.models.clone_model(teacher_model)
        optimized_model.set_weights(teacher_model.get_weights())

    # --- D. PHASE 2-5: OPTIMIZATION PIPELINE ---
    compile_config = {
        "loss_fn":   tf.keras.losses.CategoricalCrossentropy(),
        "optimizer": tf.keras.optimizers.Adam(learning_rate=0.001),
        "metrics":   ["accuracy"],
    }

    strategy = OptimizationStrategy(optimized_model, compile_config)

    target_sparsity = 0.85 if args.prune else 0.0
    quant_threshold = 0.90

    # Stem of the model file (e.g. "wrn_40_2_teacher" or "resnet110_optimized")
    # included in the filename so teacher and student outputs never collide.
    model_stem = os.path.splitext(os.path.basename(args.model_path))[0]
    timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.prune or args.quantize:
        output_filename = os.path.join(
            OUTPUTS_DIR,
            f"model_{args.dataset}_{model_stem}_{args.distill}_{args.pruning_mode}_{timestamp}.cpp"
        )
        dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(64)

        opt_graph, final_model = strategy.execute_pipeline(
            train_ds=dataset_train,
            val_x=x_test,
            val_y=y_test,
            sparsity_target=target_sparsity,
            quant_threshold=quant_threshold,
            output_name=output_filename,
            pruning_mode=args.pruning_mode,
        )
        optimized_model = final_model
    else:
        print("\n>>> Skipping Pruning/Quantization (Compilation Only)")
        from graph_optimizer import GraphExplorer, GraphOptimizer
        from compiler import Compiler

        explorer  = GraphExplorer(optimized_model)
        ir_nodes  = explorer.build_ir()
        opt_graph = GraphOptimizer(ir_nodes).optimize()

        output_filename = os.path.join(
            OUTPUTS_DIR,
            f"model_{args.dataset}_{model_stem}_baseline_{timestamp}.cpp"
        )
        Compiler(opt_graph).compile(output_filename)

    # --- E. EVALUATION ---
    print(f"\n>>> Final Evaluation")
    evaluator = PerformanceEvaluator(teacher_model, optimized_model, (x_test, y_test))
    evaluator.evaluate()


if __name__ == "__main__":
    main()
