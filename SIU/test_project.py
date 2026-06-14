"""
test_project.py  —  The Inference Compiler test runner
=======================================================
All output is streamed in real-time to:
  - stdout (console)
  - logs/experiment_YYYYMMDD_HHMMSS.log

New test groups added:
  Group C : Pruning ablation  (standard  vs  Sensitivity-Aware Recovery)
  Group D : TFLite benchmark  (Action 4.1 / 4.2 / 4.3)
"""

import os
import subprocess
import sys
import datetime

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
ROOT     = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# One timestamped log file per run
_timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_LOG = os.path.join(LOGS_DIR, f"experiment_{_timestamp}.log")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_test_streaming(test_name, flags_list, script="main.py"):
    """
    Runs a test and streams output to both console and log file in real-time.
    `script` lets us call tflite_benchmark.py without a separate helper.
    """
    full_command = [sys.executable, os.path.join(ROOT, script)] + flags_list
    cmd_str      = " ".join(full_command)

    header = (
        f"\n{'='*60}\n"
        f"TEST: {test_name}\n"
        f"TIME: {datetime.datetime.now()}\n"
        f"CMD:  {cmd_str}\n"
        f"{'='*60}\n"
    )

    print(header)
    with open(RESULTS_LOG, "a", encoding="utf-8") as f:
        f.write(header)

    # Force UTF-8 in the child process (Windows defaults to cp1252 which
    # causes UnicodeDecodeError on Keras progress-bar characters like 0x97)
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"

    process = subprocess.Popen(
        full_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",   # replace undecodable bytes with ? instead of crashing
        bufsize=1,
        env=child_env,
    )

    with open(RESULTS_LOG, "a", encoding="utf-8") as f:
        for line in process.stdout:
            print(line, end="")
            f.write(line)

    process.wait()

    status  = "PASS" if process.returncode == 0 else f"FAIL (Code: {process.returncode})"
    footer  = f"\n[STATUS] {status}\n"
    print(footer)
    with open(RESULTS_LOG, "a", encoding="utf-8") as f:
        f.write(footer)


# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
MODELS_DIR    = os.path.join(ROOT, "models")
TEACHER_MODEL = os.path.join(MODELS_DIR, "wrn_40_2_teacher.h5")
STUDENT_MODEL = os.path.join(MODELS_DIR, "resnet110_optimized.h5")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_header = f"=== TEST SUITE STARTED: {datetime.datetime.now()} ===\n"
    print(run_header)
    with open(RESULTS_LOG, "a", encoding="utf-8") as f:
        f.write(run_header)

    # =========================================================
    # PART 1: MLP TRIALS (MNIST)  — kept but commented out by default
    # =========================================================
    """
    print("\n>>> STARTING MLP TRIALS (MNIST)...")
    base_mlp_args = ["--dataset", "mnist", "--model_type", "mlp", "--epochs", "1"]

    run_test_streaming("MLP | Baseline",
                       base_mlp_args)
    run_test_streaming("MLP | No Distill + Prune + Quant",
                       base_mlp_args + ["--prune", "--quantize"])
    run_test_streaming("MLP | Our RKD + Prune + Quant",
                       base_mlp_args + ["--distill", "ours", "--prune", "--quantize"])
    run_test_streaming("MLP | External + Prune + Quant",
                       base_mlp_args + ["--distill", "external_hint", "--prune", "--quantize"])
    """
    '''
    # =========================================================
    # PART 2: RESNET TRIALS — Group A: Teacher (WideResNet-40-2)
    # =========================================================
    print("\n>>> STARTING RESNET TRIALS (CIFAR-100)...")

    base_resnet_args = [
        "--dataset",    "cifar100",
        "--model_type", "resnet",
        "--epochs",     "5",
    ]

    if os.path.exists(TEACHER_MODEL):
        print(f"\n> Group A: Teacher Benchmark ({TEACHER_MODEL})")

        run_test_streaming(
            "Teacher | Prune + Quant Only (Recovery)",
            base_resnet_args + [
                "--model_path", TEACHER_MODEL,
                "--prune", "--quantize",
                "--pruning_mode", "recovery",
            ],
        )
    else:
        print(f"\n!!! Skipping Group A: {TEACHER_MODEL} not found.")
        print("    Run 'python build_wrn_h5.py' to generate it.")

    # =========================================================
    # PART 2: RESNET TRIALS — Group B: Student (ResNet-110)
    # =========================================================
    if os.path.exists(STUDENT_MODEL):
        print(f"\n> Group B: Student Experiments ({STUDENT_MODEL})")
        student_args = base_resnet_args + ["--model_path", STUDENT_MODEL]

        run_test_streaming(
            "ResNet Student | No Distill + Prune + Quant (Recovery)",
            student_args + ["--prune", "--quantize", "--pruning_mode", "recovery"],
        )

        """
        run_test_streaming(
            "ResNet Student | Our RKD + Prune + Quant",
            student_args + ["--distill", "ours", "--prune", "--quantize",
                            "--pruning_mode", "recovery"],
        )
        run_test_streaming(
            "ResNet Student | External + Prune + Quant",
            student_args + ["--distill", "external_hint", "--prune", "--quantize",
                            "--pruning_mode", "recovery"],
        )
        run_test_streaming(
            "ResNet Student | Our RKD Only (No Prune/Quant)",
            student_args + ["--distill", "ours"],
        )
        """
    else:
        print(f"\n!!! Skipping Group B: {STUDENT_MODEL} not found.")
        print("    Run 'python build_resnet_h5.py' to generate it.")

    # =========================================================
    # PART 3: PRUNING ABLATION — Standard vs Sensitivity-Aware Recovery
    # =========================================================
    print("\n>>> STARTING PRUNING ABLATION (Standard vs Recovery)...")

    ablation_model = STUDENT_MODEL if os.path.exists(STUDENT_MODEL) else (
        TEACHER_MODEL if os.path.exists(TEACHER_MODEL) else None
    )

    if ablation_model:
        ablation_base = base_resnet_args + [
            "--model_path", ablation_model,
            "--prune", "--quantize",
        ]

        run_test_streaming(
            "Ablation | Standard Pruning (no recovery)",
            ablation_base + ["--pruning_mode", "standard"],
        )

        run_test_streaming(
            "Ablation | Sensitivity-Aware Recovery Pruning",
            ablation_base + ["--pruning_mode", "recovery"],
        )
    else:
        print("\n!!! Skipping pruning ablation: no ResNet model found.")
        print("    Provide wrn_40_2_teacher.h5 or resnet110_optimized.h5 in models/")
    '''
    # =========================================================
    # PART 4: TFLITE BENCHMARK  (Actions 4.1 / 4.2 / 4.3)
    # =========================================================
    print("\n>>> STARTING TFLITE BENCHMARK...")

    benchmark_model = STUDENT_MODEL

    #benchmark_model = TEACHER_MODEL if os.path.exists(TEACHER_MODEL) else (
    #    STUDENT_MODEL if os.path.exists(STUDENT_MODEL) else None
    #)

    if benchmark_model:
        run_test_streaming(
            "TFLite Benchmark | Python vs TFLite vs Our C++",
            ["--model_path", benchmark_model, "--cpp_latency_ms", "0.0"],
            script="tflite_benchmark.py",
        )
    else:
        print("\n!!! Skipping TFLite benchmark: no model file found in models/")
        print("    Provide wrn_40_2_teacher.h5 or resnet110_optimized.h5 in models/")

    # ---------------------------------------------------------------------------
    run_footer = f"\n=== TEST SUITE FINISHED: {datetime.datetime.now()} ===\n"
    run_footer += f"    Log saved to: {RESULTS_LOG}\n"
    print(run_footer)
    with open(RESULTS_LOG, "a", encoding="utf-8") as f:
        f.write(run_footer)