#!/usr/bin/env python3

"""
GPU Detection Script - Check available GPUs
"""

import torch
import os

print("="*60)
print("🔍 GPU DETECTION REPORT")
print("="*60)

# Check PyTorch CUDA availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"Number of GPUs detected by PyTorch: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        gpu_props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {gpu_props.name}")
        print(f"    Memory: {gpu_props.total_memory / 1024**3:.1f} GB")
        print(f"    Compute Capability: {gpu_props.major}.{gpu_props.minor}")
else:
    print("❌ No CUDA GPUs detected")

# Check environment variables
print(f"\nEnvironment variables:")
print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"  WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'Not set')}")
print(f"  LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'Not set')}")

# Check if script was launched with accelerate/torchrun
print(f"\nLaunching method detection:")
if 'RANK' in os.environ:
    print("  ✅ Launched with distributed launcher (torchrun/accelerate)")
    print(f"    RANK: {os.environ.get('RANK')}")
    print(f"    WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
    print(f"    LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
else:
    print("  ❌ NOT launched with distributed launcher")
    print("  💡 For true multi-GPU, use: accelerate launch or torchrun")

print("="*60)