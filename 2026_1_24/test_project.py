import os
import subprocess
import sys
import datetime

sys.stdout.reconfigure(line_buffering=True)

# Output file for logs
RESULTS_FILE = "experiment_results.txt"

def run_test_streaming(test_name, flags_list):
    """
    Runs a test and streams output to BOTH Console and File in real-time.
    Uses subprocess.Popen to avoid buffering issues.
    """
    full_command = [sys.executable, "main.py"] + flags_list
    cmd_str = " ".join(full_command)
    
    header = f"\n{'='*60}\nTEST: {test_name}\nTIME: {datetime.datetime.now()}\nCMD: {cmd_str}\n{'='*60}\n"
    
    print(header)
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(header)

    process = subprocess.Popen(
        full_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8', 
        bufsize=1 
    )

    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        for line in process.stdout:
            print(line, end='') 
            f.write(line)
    
    process.wait()
    
    footer = f"\n[STATUS] {'PASS' if process.returncode == 0 else 'FAIL (Code: ' + str(process.returncode) + ')'}\n"
    print(footer)
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(footer)

if __name__ == "__main__":
    print(f"=== TEST SUITE STARTED: {datetime.datetime.now()} ===\n")
    '''
    # =========================================================
    # PART 1: MLP TRIALS (MNIST)
    # =========================================================
    print("\n>>> STARTING MLP TRIALS (MNIST)...")
    
    base_mlp_args = ["--dataset", "mnist", "--model_type", "mlp", "--epochs", "1"]
    
    # Test 1: Baseline
    run_test_streaming("MLP | Baseline", 
                       base_mlp_args)

    # Test 2: Standard Pruning + Quantization (No Distill)
    run_test_streaming("MLP | No Distill + Prune + Quant", 
                       base_mlp_args + ["--prune", "--quantize"])

    # Test 3: Our Method (RKD)
    run_test_streaming("MLP | Our RKD + Prune + Quant", 
                       base_mlp_args + ["--distill", "ours", "--prune", "--quantize"])

    # Test 4: External Method (RESTORED)
    run_test_streaming("MLP | External + Prune + Quant", 
                       base_mlp_args + ["--distill", "external_hint", "--prune", "--quantize"])
    '''
    # =========================================================
    # PART 2: RESNET TRIALS (CIFAR-100)
    # =========================================================
    print("\n>>> STARTING RESNET TRIALS (CIFAR-100)...")
    
    TEACHER_MODEL = "wrn_40_2_teacher.h5"
    STUDENT_MODEL = "resnet110_optimized.h5"
    
    # Using 5 epochs for ResNet recovery
    base_resnet_args = ["--dataset", "cifar100", "--model_type", "resnet", "--epochs", "5"]
    
    # ---------------------------------------------------------
    # GROUP A: TEACHER BENCHMARKS (WideResNet-40-2)
    # ---------------------------------------------------------
    if os.path.exists(TEACHER_MODEL):
        print(f"\n> Group A: Teacher Benchmark ({TEACHER_MODEL})")
        
        # Test 1: Teacher Baseline
        #run_test_streaming("Teacher | Baseline (No Opt)", 
         #                  base_resnet_args + ["--model_path", TEACHER_MODEL])
        
        # Test 2: Teacher + Prune + Quant Only
        run_test_streaming("Teacher | Prune + Quant Only", 
                           base_resnet_args + ["--model_path", TEACHER_MODEL, "--prune", "--quantize"])
    else:
        print(f"\n!!! Skipping Group A: Teacher Model {TEACHER_MODEL} not found.")
        print("    (Run 'python build_wrn_h5.py' to generate it)")
    
    # ---------------------------------------------------------
    # GROUP B: STUDENT EXPERIMENTS (ResNet-110)
    # ---------------------------------------------------------
    if os.path.exists(STUDENT_MODEL):
        print(f"\n> Group B: Student Experiments ({STUDENT_MODEL})")
        
        student_args = base_resnet_args + ["--model_path", STUDENT_MODEL]

        # Test 1: Baseline
        #run_test_streaming("ResNet Student | Baseline", 
         #                  student_args)

        # Test 2: No Distill (Standard Pruning)
        run_test_streaming("ResNet Student | No Distill + Prune + Quant", 
                           student_args + ["--prune", "--quantize"])
        '''
        # Test 3: Our Distill (RKD + Prune + Quant)
        run_test_streaming("ResNet Student | Our RKD + Prune + Quant", 
                           student_args + ["--distill", "ours", "--prune", "--quantize"])

        # Test 4: External Distill (Reference Method)
        run_test_streaming("ResNet Student | External + Prune + Quant", 
                           student_args + ["--distill", "external_hint", "--prune", "--quantize"])

        # Test 5: Our Distill Only (No Compression - Just Finetuning)
        run_test_streaming("ResNet Student | Our RKD Only (No Prune/Quant)", 
                           student_args + ["--distill", "ours"])'''
    else:
        print(f"\n!!! Skipping Group B: Student Model {STUDENT_MODEL} not found.")
        print("    (Run 'python build_resnet_h5.py' to generate it)")