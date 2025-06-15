#!/usr/bin/env python3

"""
Multi-GPU Training Verification Script
Monitors and confirms that multi-GPU training is actually utilizing multiple GPUs
"""

import subprocess
import time
import json
import psutil
from datetime import datetime

def check_gpu_usage():
    """Check current GPU usage"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(', ')
                gpu_info.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'utilization': int(parts[2]),
                    'memory_used': int(parts[3]),
                    'memory_total': int(parts[4])
                })
        return gpu_info
    except:
        return []

def monitor_training(duration_seconds=300):
    """Monitor training for specified duration"""
    print("Multi-GPU Training Verification Monitor")
    print("="*50)
    print(f"Monitoring for {duration_seconds} seconds...")
    print("Looking for signs of multi-GPU utilization...")
    print("")
    
    start_time = time.time()
    samples = []
    
    while time.time() - start_time < duration_seconds:
        gpu_info = check_gpu_usage()
        
        if gpu_info:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] GPU Status:")
            
            active_gpus = 0
            total_utilization = 0
            
            for gpu in gpu_info:
                utilization = gpu['utilization']
                memory_used = gpu['memory_used']
                memory_total = gpu['memory_total']
                memory_pct = (memory_used / memory_total) * 100
                
                status = "ACTIVE" if utilization > 10 else "IDLE"
                if utilization > 10:
                    active_gpus += 1
                    total_utilization += utilization
                
                print(f"  GPU {gpu['index']}: {status} - {utilization}% util, {memory_pct:.1f}% memory ({memory_used}MB/{memory_total}MB)")
            
            # Record sample
            sample = {
                'timestamp': timestamp,
                'active_gpus': active_gpus,
                'avg_utilization': total_utilization / len(gpu_info) if gpu_info else 0,
                'gpus': gpu_info
            }
            samples.append(sample)
            
            # Multi-GPU detection
            if active_gpus > 1:
                print(f"  ✅ MULTI-GPU DETECTED: {active_gpus} GPUs active!")
            elif active_gpus == 1:
                print(f"  ⚠️  SINGLE GPU: Only 1 GPU active")
            else:
                print(f"  ❌ NO GPU ACTIVITY")
            
            print("")
        
        time.sleep(5)
    
    # Summary
    print("\nMONITORING SUMMARY")
    print("="*30)
    
    if samples:
        multi_gpu_samples = [s for s in samples if s['active_gpus'] > 1]
        single_gpu_samples = [s for s in samples if s['active_gpus'] == 1]
        idle_samples = [s for s in samples if s['active_gpus'] == 0]
        
        print(f"Total samples: {len(samples)}")
        print(f"Multi-GPU active: {len(multi_gpu_samples)} samples ({len(multi_gpu_samples)/len(samples)*100:.1f}%)")
        print(f"Single GPU active: {len(single_gpu_samples)} samples ({len(single_gpu_samples)/len(samples)*100:.1f}%)")
        print(f"No GPU activity: {len(idle_samples)} samples ({len(idle_samples)/len(samples)*100:.1f}%)")
        
        if multi_gpu_samples:
            avg_utilization = sum(s['avg_utilization'] for s in multi_gpu_samples) / len(multi_gpu_samples)
            print(f"Average GPU utilization during multi-GPU: {avg_utilization:.1f}%")
            print("✅ CONFIRMED: Multi-GPU training is working!")
        else:
            print("❌ WARNING: No multi-GPU activity detected")
    else:
        print("No GPU data collected")

def check_training_processes():
    """Check for running Python training processes"""
    print("ACTIVE TRAINING PROCESSES")
    print("="*30)
    
    training_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] == 'python' and proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline'])
                if 'train_multi_gpu.py' in cmdline:
                    training_processes.append(proc.info)
        except:
            continue
    
    if training_processes:
        print(f"Found {len(training_processes)} training process(es):")
        for proc in training_processes:
            print(f"  PID {proc['pid']}: {' '.join(proc['cmdline'])}")
        print("✅ Training processes detected")
    else:
        print("❌ No training processes found")
        print("💡 Start training in another terminal, then run this monitor")
    
    return len(training_processes) > 0

if __name__ == "__main__":
    import sys
    
    # Check if training is running
    if not check_training_processes():
        print("\nTo use this monitor:")
        print("1. Start training in another terminal:")
        print("   accelerate launch train_multi_gpu.py --trainsession v1 --config medium")
        print("2. Run this monitor:")
        print("   python verify_multi_gpu.py")
        sys.exit(1)
    
    print("")
    
    # Monitor training
    duration = 120  # Monitor for 2 minutes
    monitor_training(duration)