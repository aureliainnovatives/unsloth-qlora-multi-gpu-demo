#!/bin/bash

# Setup script for Accelerate configuration
echo "🚀 Setting up Accelerate for Multi-GPU Training"
echo "================================================"

# Check if accelerate is installed
if ! command -v accelerate &> /dev/null; then
    echo "❌ Accelerate not found. Please install with: pip install accelerate"
    exit 1
fi

# Check GPU availability
echo "🔍 Checking GPU availability..."
python -c "
import torch
gpu_count = torch.cuda.device_count()
print(f'Available GPUs: {gpu_count}')
if gpu_count == 0:
    print('❌ No GPUs found!')
    exit(1)
elif gpu_count == 1:
    print('⚠️  Only 1 GPU found. Multi-GPU training not possible.')
else:
    print(f'✅ {gpu_count} GPUs available for multi-GPU training')
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        print(f'   GPU {i}: {props.name} ({memory_gb:.1f}GB)')
"

echo ""
echo "📝 Setting up Accelerate configuration..."

# Copy our pre-configured accelerate config
ACCELERATE_CONFIG_DIR="$HOME/.cache/huggingface/accelerate"
mkdir -p "$ACCELERATE_CONFIG_DIR"

# Update num_processes based on available GPUs
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
if [ "$GPU_COUNT" -gt 1 ]; then
    # Create config with detected GPU count
    cat > "$ACCELERATE_CONFIG_DIR/default_config.yaml" << EOF
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
enable_cpu_affinity: false
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: $GPU_COUNT
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
    echo "✅ Multi-GPU config created for $GPU_COUNT GPUs"
else
    # Single GPU config
    cat > "$ACCELERATE_CONFIG_DIR/default_config.yaml" << EOF
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: 'NO'
downcast_bf16: 'no'
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
use_cpu: false
EOF
    echo "✅ Single GPU config created"
fi

echo ""
echo "🔧 Verifying Accelerate configuration..."
accelerate env

echo ""
echo "✅ Accelerate setup complete!"
echo ""
echo "📋 Next steps:"
echo "   Single GPU:   python train.py --force-single-gpu"
echo "   Multi-GPU:    accelerate launch train.py"
echo "   Monitor:      python monitor_gpu.py"
echo ""