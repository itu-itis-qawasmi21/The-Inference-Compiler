# The Inference Compiler

High-performance neural network compression and C++ code generation pipeline.
Supports structured/unstructured pruning, quantization, graph optimization,
and direct C++ backend compilation with OpenMP parallelism.

---

## Directory Layout

```
TheInferenceCompiler/
│
├── src/                        # All importable Python modules
│   ├── compiler.py             # C++ code generator (sparse unrolling, OpenMP)
│   ├── evaluator.py            # Accuracy / sparsity / latency evaluation
│   ├── graph_optimizer.py      # IR builder + DCE / CSE / BN-fold / fusion passes
│   ├── pruning.py              # PruningEngine (Greedy → Heuristic → Weights)
│   ├── quantization.py         # QuantizationEngine (sensitivity-aware QAT)
│   └── strategy_manager.py     # Pipeline orchestrator
│
├── models/                     # Pre-trained Keras .h5 models (not committed)
│   ├── wrn_40_2_teacher.h5
│   └── resnet110_optimized.h5
│
├── outputs/                    # Generated C++ files (auto-created)
│   └── model_cifar100_none_recovery.cpp
│
├── logs/                       # Timestamped run logs (auto-created)
│   └── experiment_YYYYMMDD_HHMMSS.log
│
├── results/                    # JSON benchmark sidecars (auto-created)
│   └── benchmark_results.json
│
├── main.py                     # Main pipeline entry point
├── tflite_benchmark.py         # TFLite conversion + latency comparison table
└── test_project.py             # Full test suite runner
```

---

## Running the Test Suite

```bash
cd TheInferenceCompiler
python test_project.py
```

Logs are written to `logs/experiment_<timestamp>.log` automatically.

---

## Test Groups

| Group | Description |
|-------|-------------|
| A     | Teacher model (WideResNet-40-2) — Prune + Quant |
| B     | Student model (ResNet-110) — Prune + Quant |
| C     | **Pruning ablation** — `standard` vs `recovery` mode on CIFAR-100 |
| D     | **TFLite benchmark** — Python → TFLite → Our C++ latency table |

---

## Key Arguments (`main.py`)

| Flag | Values | Description |
|------|--------|-------------|
| `--dataset` | `mnist`, `cifar100` | Dataset to use |
| `--model_type` | `mlp`, `resnet` | Architecture family |
| `--model_path` | path | Path to `.h5` file inside `models/` |
| `--distill` | `none`, `ours`, `external_hint` | Distillation method |
| `--prune` | flag | Enable pruning phase |
| `--quantize` | flag | Enable quantization phase |
| `--pruning_mode` | `recovery` *(default)*, `standard` | Phase 2 behaviour |
| `--epochs` | int | Distillation epochs |

### `--pruning_mode` explained

- **`recovery`** — full Sensitivity-Aware Recovery: after each failed pruning
  step, neurons are triaged into *guilty* (protected for 15 steps) and
  *innocent* (pruned immediately). This is the method being presented.
- **`standard`** — no recovery logic; Phase 2 terminates as soon as the
  accuracy budget is exhausted. Used as the ablation baseline in the
  results table.

---

## TFLite Benchmark (`tflite_benchmark.py`)

Converts the Keras model to TFLite with `DEFAULT` optimizations, measures
batch-32 equivalent latency over 50 iterations, and prints:

```
Python / Keras (baseline)        XX.XX ms      1.00x
TFLite (DEFAULT opt.)             X.XX ms      X.XXx
Our C++ Compiler                  0.XXX ms    XXX.XXx
```

Results are saved to `results/benchmark_results.json`.

To update the C++ latency once you have a measured number, pass it directly:

```bash
python tflite_benchmark.py --model_path models/wrn_40_2_teacher.h5 \
                           --cpp_latency_ms 0.077
```
