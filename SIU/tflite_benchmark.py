"""
tflite_benchmark.py
-------------------
Converts a Keras .h5 model to TFLite (with DEFAULT optimizations),
runs timed inference on CIFAR-100 test data, and writes results to
results/benchmark_results.json.

Also prints the full comparison table:
  Python Baseline  →  TFLite  →  Our C++ Compiler

Usage (called from test_project.py via subprocess):
    python tflite_benchmark.py --model_path models/wrn_40_2_teacher.h5
                               --cpp_latency_ms 0.077
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import tensorflow as tf

# Allow running from project root
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(RESULTS_DIR, "benchmark_results.json")

TFLITE_MODEL_PATH = os.path.join(ROOT, "models", "benchmark_model.tflite")

ITERATIONS = 50
BATCH_SIZE  = 32


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_cifar100_test():
    print("--- [Benchmark] Loading CIFAR-100 test set ---")
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    x_test = x_test.astype("float32") / 255.0
    y_test = tf.keras.utils.to_categorical(y_test, 100)
    return x_test, y_test


# ---------------------------------------------------------------------------
# Python baseline latency (Keras model, batch=32)
# ---------------------------------------------------------------------------
def measure_python_latency(model, x_test):
    print("--- [Benchmark] Measuring Python (Keras) latency ---")
    # Warmup
    _ = model(x_test[:BATCH_SIZE], training=False)

    start = time.time()
    for _ in range(ITERATIONS):
        _ = model(x_test[:BATCH_SIZE], training=False)
    elapsed = time.time() - start

    ms = (elapsed / ITERATIONS) * 1000
    print(f"  Python Baseline : {ms:.2f} ms  (avg over {ITERATIONS} iters, batch={BATCH_SIZE})")
    return ms


# ---------------------------------------------------------------------------
# TFLite conversion + latency
# ---------------------------------------------------------------------------
def convert_to_tflite(model):
    print("--- [Benchmark] Converting to TFLite (DEFAULT optimizations) ---")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]   # fp16 + weight quantisation
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(TFLITE_MODEL_PATH), exist_ok=True)
    with open(TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"  TFLite model saved → {TFLITE_MODEL_PATH}  ({size_kb:.1f} KB)")
    return tflite_model


def measure_tflite_latency(x_test):
    """
    Runs TFLite interpreter on single samples (TFLite doesn't batch easily
    without dynamic shapes), averaged over ITERATIONS × BATCH_SIZE inferences,
    then scaled to a batch-32 equivalent for a fair comparison.
    """
    print("--- [Benchmark] Measuring TFLite latency ---")
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Check whether the model accepts batches or single samples
    input_shape = input_details[0]["shape"]  # e.g. [1, 32, 32, 3]
    single_input = input_shape[0] == 1

    # Warmup
    sample = x_test[:1].astype(np.float32)
    interpreter.set_tensor(input_details[0]["index"], sample)
    interpreter.invoke()

    start = time.time()
    n_inferences = ITERATIONS * BATCH_SIZE if single_input else ITERATIONS
    for i in range(n_inferences):
        if single_input:
            inp = x_test[i % len(x_test) : i % len(x_test) + 1].astype(np.float32)
        else:
            inp = x_test[:BATCH_SIZE].astype(np.float32)
        interpreter.set_tensor(input_details[0]["index"], inp)
        interpreter.invoke()
    elapsed = time.time() - start

    if single_input:
        # time for BATCH_SIZE single inferences = one "batch"
        ms_per_batch = (elapsed / ITERATIONS) * 1000
    else:
        ms_per_batch = (elapsed / ITERATIONS) * 1000

    print(f"  TFLite          : {ms_per_batch:.2f} ms  (batch-32 equivalent, {ITERATIONS} iters)")
    return ms_per_batch


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------
def print_comparison_table(python_ms, tflite_ms, cpp_ms):
    print("\n" + "=" * 65)
    print("  LATENCY COMPARISON TABLE  (AMD Ryzen CPU, Batch=32, CIFAR-100)")
    print("=" * 65)
    print(f"  {'Backend':<28} {'Latency (ms)':>14}  {'Speedup vs Python':>18}")
    print("-" * 65)

    rows = [
        ("Python / Keras (baseline)",  python_ms,  1.0),
        ("TFLite (DEFAULT opt.)",       tflite_ms,  python_ms / max(tflite_ms, 1e-9)),
        ("Our C++ Compiler",            cpp_ms,     python_ms / max(cpp_ms,    1e-9)),
    ]
    for name, ms, speedup in rows:
        print(f"  {name:<28} {ms:>12.2f}   {speedup:>14.2f}x")

    print("=" * 65)
    print(f"\n  Pipeline: Python ({python_ms:.2f} ms)"
          f"  →  TFLite ({tflite_ms:.2f} ms)"
          f"  →  Our C++ ({cpp_ms:.2f} ms)\n")


# ---------------------------------------------------------------------------
# Persist results
# ---------------------------------------------------------------------------
def save_results(python_ms, tflite_ms, cpp_ms):
    data = {}
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE) as f:
                data = json.load(f)
        except Exception:
            data = {}

    data["tflite_benchmark"] = {
        "python_baseline_ms": round(python_ms, 4),
        "tflite_ms":          round(tflite_ms, 4),
        "cpp_compiler_ms":    round(cpp_ms, 4),
        "tflite_speedup":     round(python_ms / max(tflite_ms, 1e-9), 2),
        "cpp_speedup":        round(python_ms / max(cpp_ms,    1e-9), 2),
        "batch_size":         BATCH_SIZE,
        "iterations":         ITERATIONS,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Results saved → {RESULTS_FILE}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TFLite benchmark for The Inference Compiler")
    parser.add_argument("--model_path",    type=str, required=True,
                        help="Path to .h5 Keras model (e.g. models/wrn_40_2_teacher.h5)")
    parser.add_argument("--cpp_latency_ms", type=float, default=None,
                        help="Pre-measured C++ compiler latency in ms (optional). "
                             "If omitted, a placeholder of 0.0 is used in the table.")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"!!! ERROR: Model not found: {args.model_path}")
        sys.exit(1)

    # 1. Load data + model
    x_test, y_test = load_cifar100_test()
    print(f"--- [Benchmark] Loading model: {args.model_path} ---")
    model = tf.keras.models.load_model(args.model_path)

    # 2. Python baseline
    python_ms = measure_python_latency(model, x_test)

    # 3. TFLite
    convert_to_tflite(model)
    tflite_ms = measure_tflite_latency(x_test)

    # 4. C++ latency (passed in from test runner or left as placeholder)
    cpp_ms = args.cpp_latency_ms if args.cpp_latency_ms is not None else 0.0

    # 5. Print table + save
    print_comparison_table(python_ms, tflite_ms, cpp_ms)
    save_results(python_ms, tflite_ms, cpp_ms)


if __name__ == "__main__":
    main()
