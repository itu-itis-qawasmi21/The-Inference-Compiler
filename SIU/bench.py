import tensorflow as tf
import time
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar100'])
    args = parser.parse_args()

    # 1. Load Model
    print(f"Loading model: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)

    # 2. Create Dummy Data (Batch Size 1)
    if args.dataset == 'mnist':
        dummy_input = np.random.rand(1, 784).astype(np.float32)
    else:
        dummy_input = np.random.rand(1, 32, 32, 3).astype(np.float32)

    # 3. THE WARM-UP PHASE (Crucial for fair comparison)
    print("--- Warming up TensorFlow Graph ---")
    for _ in range(50):
        _ = model(dummy_input, training=False)

    # 4. THE BENCHMARK
    iterations = 1000
    print(f"--- Starting Python Benchmark ({iterations} iterations) ---")
    
    # time.perf_counter is highly accurate for microsecond CPU timings
    start_time = time.perf_counter() 
    
    for _ in range(iterations):
        _ = model(dummy_input, training=False)
        
    end_time = time.perf_counter()

    total_time_ms = (end_time - start_time) * 1000
    avg_latency = total_time_ms / iterations

    print("==========================================")
    print(f"Total Execution Time: {total_time_ms:.3f} ms")
    print(f"Average Latency:      {avg_latency:.3f} ms")
    print("==========================================")

if __name__ == "__main__":
    main()