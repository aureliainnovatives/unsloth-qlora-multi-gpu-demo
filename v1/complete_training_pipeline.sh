#!/bin/bash

# Complete Multi-GPU QLoRA Training Pipeline
# This script sets up the environment and runs comprehensive training experiments

set -e  # Exit on any error

echo "=========================================="
echo "Multi-GPU QLoRA Training Pipeline"
echo "=========================================="
echo "This script will:"
echo "1. Set up Python virtual environment"
echo "2. Install required dependencies"
echo "3. Run training experiments (single & multi-GPU)"
echo "4. Monitor GPU usage during training"
echo "5. Analyze and compare results"
echo ""

# Configuration
VENV_NAME="qlora_env"
SESSION_NAME="${1:-v1}"  # Default session name is v1
CONFIG_SIZE="${2:-small}"  # Default config size is small

echo "Session Name: $SESSION_NAME"
echo "Configuration: $CONFIG_SIZE"
echo ""

# Step 1: Virtual Environment Setup
echo "Step 1: Setting up virtual environment..."
echo "----------------------------------------"

if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment: $VENV_NAME"
    python3 -m venv $VENV_NAME
else
    echo "Virtual environment already exists: $VENV_NAME"
fi

echo "Activating virtual environment..."
source $VENV_NAME/bin/activate

# Verify Python version
echo "Python version: $(python --version)"
echo ""

# Step 2: Dependencies Installation
echo "Step 2: Installing dependencies..."
echo "--------------------------------"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installing core dependencies..."
pip install transformers accelerate datasets tokenizers

echo "Installing PEFT and LoRA support..."
pip install peft

echo "Installing quantization support..."
pip install bitsandbytes

echo "Installing TRL for supervised fine-tuning..."
pip install trl

echo "Installing Unsloth (may take a few minutes)..."
pip install unsloth[colab-new] || echo "Unsloth installation failed - continuing without it"

echo "Installing monitoring and visualization tools..."
pip install psutil gpustat matplotlib seaborn

echo "Installing additional utilities..."
pip install wandb tensorboard

echo ""
echo "Dependencies installation completed!"
echo ""

# Step 3: Accelerate Configuration
echo "Step 3: Configuring Accelerate..."
echo "--------------------------------"

if [ ! -f "$HOME/.cache/huggingface/accelerate/default_config.yaml" ]; then
    echo "Setting up basic accelerate configuration..."
    accelerate config default
else
    echo "Accelerate already configured"
fi

echo ""

# Step 4: GPU Detection and Verification
echo "Step 4: GPU Detection and Verification..."
echo "----------------------------------------"

echo "Running GPU detection script..."
python check_gpus.py

echo ""

# Step 5: Training Configuration Display
echo "Step 5: Training Configuration Preview..."
echo "----------------------------------------"

echo "Available training configurations:"
python training_configs.py

echo ""

# Step 6: Training Execution
echo "Step 6: Starting Training Experiments..."
echo "======================================="

echo "Starting training session: $SESSION_NAME with config: $CONFIG_SIZE"
echo ""

# Single GPU Training
echo "PHASE 1: Single GPU Training"
echo "----------------------------"
echo "Command: python train_single_gpu.py --trainsession $SESSION_NAME --config $CONFIG_SIZE"
echo ""

# Start GPU monitoring in background
echo "Starting GPU monitoring..."
gpustat -i 5 > gpu_usage_single_${SESSION_NAME}_${CONFIG_SIZE}.log &
GPU_MONITOR_PID=$!

# Run single GPU training
python train_single_gpu.py --trainsession $SESSION_NAME --config $CONFIG_SIZE

# Stop GPU monitoring
kill $GPU_MONITOR_PID 2>/dev/null || true
echo "Single GPU training completed!"
echo ""

# Multi-GPU Training
echo "PHASE 2: Multi-GPU Training"
echo "---------------------------"
echo "Command: accelerate launch train_multi_gpu.py --trainsession $SESSION_NAME --config $CONFIG_SIZE"
echo ""

# Start GPU monitoring in background
echo "Starting GPU monitoring..."
gpustat -i 5 > gpu_usage_multi_${SESSION_NAME}_${CONFIG_SIZE}.log &
GPU_MONITOR_PID=$!

# Run multi-GPU training
accelerate launch train_multi_gpu.py --trainsession $SESSION_NAME --config $CONFIG_SIZE

# Stop GPU monitoring
kill $GPU_MONITOR_PID 2>/dev/null || true
echo "Multi-GPU training completed!"
echo ""

# Step 7: Results Analysis
echo "Step 7: Training Results Analysis..."
echo "===================================="

echo "Analyzing training results for session: $SESSION_NAME"
python analyze_training_results.py --session $SESSION_NAME

echo ""

# Step 8: Log Analysis
echo "Step 8: Detailed Log Analysis..."
echo "==============================="

echo "Generating detailed log analysis and visualizations..."
python log_viewer.py --compare --plot

echo ""

# Step 9: GPU Usage Analysis
echo "Step 9: GPU Usage Analysis..."
echo "============================"

echo "Single GPU Usage Log:"
echo "----------------------"
if [ -f "gpu_usage_single_${SESSION_NAME}_${CONFIG_SIZE}.log" ]; then
    echo "Log file: gpu_usage_single_${SESSION_NAME}_${CONFIG_SIZE}.log"
    echo "Last 10 lines:"
    tail -10 gpu_usage_single_${SESSION_NAME}_${CONFIG_SIZE}.log
else
    echo "No single GPU usage log found"
fi

echo ""
echo "Multi-GPU Usage Log:"
echo "--------------------"
if [ -f "gpu_usage_multi_${SESSION_NAME}_${CONFIG_SIZE}.log" ]; then
    echo "Log file: gpu_usage_multi_${SESSION_NAME}_${CONFIG_SIZE}.log"
    echo "Last 10 lines:"
    tail -10 gpu_usage_multi_${SESSION_NAME}_${CONFIG_SIZE}.log
else
    echo "No multi-GPU usage log found"
fi

echo ""

# Step 10: Summary and Next Steps
echo "Step 10: Pipeline Summary..."
echo "==========================="

echo "Training pipeline completed successfully!"
echo ""
echo "Generated outputs:"
echo "- Training results: sessions/$SESSION_NAME/"
echo "- GPU usage logs: gpu_usage_*_${SESSION_NAME}_${CONFIG_SIZE}.log"
echo "- Performance plots: training_comparison.png (if matplotlib available)"
echo ""
echo "To run additional experiments:"
echo "1. Different config sizes:"
echo "   bash complete_training_pipeline.sh $SESSION_NAME medium"
echo "   bash complete_training_pipeline.sh $SESSION_NAME large"
echo ""
echo "2. New session:"
echo "   bash complete_training_pipeline.sh v2 small"
echo ""
echo "3. View real-time training logs:"
echo "   tail -f sessions/$SESSION_NAME/single_gpu/trainer_state.json"
echo "   tail -f sessions/$SESSION_NAME/multi_gpu/trainer_state.json"
echo ""
echo "4. Compare multiple sessions:"
echo "   python analyze_training_results.py"
echo ""
echo "5. Monitor GPU usage in real-time:"
echo "   watch -n 1 gpustat"
echo ""

echo "Virtual environment: $VENV_NAME (remember to activate it: source $VENV_NAME/bin/activate)"
echo ""
echo "Happy training!"