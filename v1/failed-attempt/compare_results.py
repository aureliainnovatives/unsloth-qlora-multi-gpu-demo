#!/usr/bin/env python3

"""
Compare Single GPU vs Multi-GPU Results
"""

import json
import os

def load_results(file_path):
    """Load results from JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def main():
    print("="*60)
    print("📊 SINGLE GPU vs MULTI-GPU COMPARISON")
    print("="*60)
    
    # Load results
    single_gpu_results = load_results("./single_gpu_output/results.json")
    multi_gpu_results = load_results("./multi_gpu_output/results.json")
    
    if not single_gpu_results:
        print("❌ Single GPU results not found. Run: python train_single_gpu.py")
        return
    
    if not multi_gpu_results:
        print("❌ Multi-GPU results not found. Run: accelerate launch train_multi_gpu.py")
        return
    
    # Display comparison
    print(f"\n🔥 SINGLE GPU RESULTS:")
    print(f"  Duration: {single_gpu_results['duration_seconds']:.1f} seconds")
    print(f"  Final Loss: {single_gpu_results['final_loss']:.4f}")
    print(f"  Speed: {single_gpu_results['steps_per_second']:.2f} steps/sec")
    print(f"  Optimizations: {single_gpu_results['optimizations']}")
    
    print(f"\n🚀 MULTI-GPU RESULTS:")
    print(f"  Duration: {multi_gpu_results['duration_seconds']:.1f} seconds")
    print(f"  Final Loss: {multi_gpu_results['final_loss']:.4f}")
    print(f"  Speed: {multi_gpu_results['steps_per_second']:.2f} steps/sec")
    print(f"  GPUs Used: {multi_gpu_results['num_gpus']}")
    print(f"  Optimizations: {multi_gpu_results['optimizations']}")
    
    # Calculate speedup
    speedup = multi_gpu_results['steps_per_second'] / single_gpu_results['steps_per_second']
    efficiency = speedup / multi_gpu_results['num_gpus'] * 100
    
    print(f"\n⚡ PERFORMANCE ANALYSIS:")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  GPU Efficiency: {efficiency:.1f}%")
    
    time_saved = single_gpu_results['duration_seconds'] - multi_gpu_results['duration_seconds']
    time_saved_percent = (time_saved / single_gpu_results['duration_seconds']) * 100
    
    print(f"  Time Saved: {time_saved:.1f} seconds ({time_saved_percent:.1f}%)")
    
    print(f"\n📋 SUMMARY:")
    if speedup > 1.5:
        print(f"  ✅ Multi-GPU shows good speedup ({speedup:.2f}x)")
    elif speedup > 1.2:
        print(f"  ⚠️  Multi-GPU shows moderate speedup ({speedup:.2f}x)")
    else:
        print(f"  ❌ Multi-GPU speedup is limited ({speedup:.2f}x)")
    
    if efficiency > 80:
        print(f"  ✅ Excellent GPU efficiency ({efficiency:.1f}%)")
    elif efficiency > 60:
        print(f"  ⚠️  Good GPU efficiency ({efficiency:.1f}%)")
    else:
        print(f"  ❌ Low GPU efficiency ({efficiency:.1f}%)")
    
    print("="*60)

if __name__ == "__main__":
    main()