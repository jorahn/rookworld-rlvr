#!/usr/bin/env python3
"""Monitor memory usage during training to detect leaks"""

import subprocess
import re
import sys

def extract_memory(line):
    """Extract memory values from log line"""
    # Pattern: GPU 0 memory - allocated: 1.89GB, reserved: 2.31GB
    match = re.search(r'GPU (\d) memory - allocated: ([\d.]+)GB, reserved: ([\d.]+)GB', line)
    if match:
        return int(match.group(1)), float(match.group(2)), float(match.group(3))
    return None

def extract_step(line):
    """Extract step number from log line"""
    # Pattern: TRAINING STEP 1/200
    match = re.search(r'TRAINING STEP (\d+)/(\d+)', line)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None

def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    
    print(f"Running {steps} training steps with batch_size={batch_size}")
    print("Step | GPU0 Alloc | GPU0 Resv | GPU1 Alloc | GPU1 Resv")
    print("-" * 60)
    
    cmd = f"uv run python train_lean.py --steps {steps} --batch-size {batch_size} --log-level INFO 2>&1"
    
    current_step = 0
    gpu0_alloc = []
    gpu0_resv = []
    gpu1_alloc = []
    gpu1_resv = []
    
    with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        for line in proc.stdout:
            # Check for step
            step_info = extract_step(line)
            if step_info:
                current_step = step_info[0]
            
            # Check for memory info
            mem_info = extract_memory(line)
            if mem_info and current_step > 0:
                gpu_id, alloc, resv = mem_info
                if gpu_id == 0:
                    gpu0_alloc.append(alloc)
                    gpu0_resv.append(resv)
                elif gpu_id == 1:
                    gpu1_alloc.append(alloc)
                    gpu1_resv.append(resv)
                
                # Print every 10 steps
                if current_step % 10 == 0 and gpu_id == 1:  # After both GPUs reported
                    print(f"{current_step:4d} | {gpu0_alloc[-1]:10.2f} | {gpu0_resv[-1]:9.2f} | {gpu1_alloc[-1]:10.2f} | {gpu1_resv[-1]:9.2f}")
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    if gpu0_alloc:
        print(f"GPU0 Allocated: Start={gpu0_alloc[0]:.2f}GB, End={gpu0_alloc[-1]:.2f}GB, Delta={gpu0_alloc[-1]-gpu0_alloc[0]:.2f}GB")
        print(f"GPU0 Reserved:  Start={gpu0_resv[0]:.2f}GB, End={gpu0_resv[-1]:.2f}GB, Delta={gpu0_resv[-1]-gpu0_resv[0]:.2f}GB")
    if gpu1_alloc:
        print(f"GPU1 Allocated: Start={gpu1_alloc[0]:.2f}GB, End={gpu1_alloc[-1]:.2f}GB, Delta={gpu1_alloc[-1]-gpu1_alloc[0]:.2f}GB")
        print(f"GPU1 Reserved:  Start={gpu1_resv[0]:.2f}GB, End={gpu1_resv[-1]:.2f}GB, Delta={gpu1_resv[-1]-gpu1_resv[0]:.2f}GB")
    
    # Check for memory leak
    if gpu0_alloc and (gpu0_alloc[-1] - gpu0_alloc[0]) > 0.5:
        print("\n⚠️  WARNING: Possible memory leak detected on GPU0!")
    elif gpu0_alloc:
        print("\n✅ No significant memory leak detected")

if __name__ == "__main__":
    main()