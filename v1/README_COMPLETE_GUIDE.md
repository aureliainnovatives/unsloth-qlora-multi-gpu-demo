# Complete Multi-GPU QLoRA Training Guide

## Overview
This guide provides a complete setup and execution pipeline for running QLoRA fine-tuning experiments comparing single GPU vs multi-GPU performance.

## Quick Start

### 1. Run Complete Pipeline (Automated)
```bash
# Run with default settings (session: v1, config: small)
bash complete_training_pipeline.sh

# Run with custom session and config
bash complete_training_pipeline.sh v2 medium

# Run large config experiment
bash complete_training_pipeline.sh production large
```

### 2. Manual Step-by-Step Execution

#### Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv qlora_env
source qlora_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate datasets peft bitsandbytes trl
pip install unsloth[colab-new]
pip install psutil gpustat matplotlib seaborn

# Configure accelerate
accelerate config
```

#### GPU Detection
```bash
# Check available GPUs
python check_gpus.py

# View training configurations
python training_configs.py
```

#### Single GPU Training
```bash
# Small config
python train_single_gpu.py --trainsession v1 --config small

# Medium config  
python train_single_gpu.py --trainsession v1 --config medium

# Large config
python train_single_gpu.py --trainsession v1 --config large
```

#### Multi-GPU Training
```bash
# Small config
accelerate launch train_multi_gpu.py --trainsession v1 --config small

# Medium config
accelerate launch train_multi_gpu.py --trainsession v1 --config medium

# Large config
accelerate launch train_multi_gpu.py --trainsession v1 --config large
```

#### Real-time GPU Monitoring
```bash
# Monitor GPU usage during training
watch -n 1 gpustat

# Log GPU usage to file
gpustat -i 5 > gpu_usage.log &
# (run your training command)
# kill background job when done
```

#### Results Analysis
```bash
# Analyze specific session
python analyze_training_results.py --session v1

# Analyze all sessions
python analyze_training_results.py

# Generate performance plots
python log_viewer.py --compare --plot

# View specific session logs
python log_viewer.py --output-dir sessions/v1/single_gpu
```

## Configuration Options

### Training Sizes
- **small**: DialoGPT-small (117M), 20 steps, 8-bit quantization
- **medium**: DialoGPT-medium (345M), 100 steps, 8-bit quantization  
- **large**: Llama-2-7B (7B), 500 steps, 4-bit quantization

### Session Management
Sessions organize your experiments in structured folders:
```
sessions/
  v1/
    single_gpu/
      results.json, trainer_state.json, final_model/
    multi_gpu/
      results.json, trainer_state.json, final_model/
  v2/
    single_gpu/
    multi_gpu/
```

## Output Structure

### Training Results
- `sessions/SESSION_NAME/single_gpu/results.json` - Single GPU metrics
- `sessions/SESSION_NAME/multi_gpu/results.json` - Multi-GPU metrics  
- `sessions/SESSION_NAME/*/trainer_state.json` - Detailed training logs
- `sessions/SESSION_NAME/*/final_model/` - Saved model files

### Analysis Outputs
- `training_comparison.png` - Performance visualization plots
- `gpu_usage_*.log` - GPU utilization logs
- Console output with detailed comparisons

## Key Metrics Analyzed

### Performance Metrics
- **Training Duration**: Total time taken
- **Steps Per Second**: Training throughput
- **GPU Utilization**: Multi-GPU scaling efficiency
- **Memory Usage**: Model and checkpoint sizes

### Quality Metrics  
- **Final Loss**: Model convergence quality
- **Loss Convergence**: Training stability
- **Model Size**: Output file sizes

### Speedup Analysis
- **Speedup Factor**: Multi-GPU vs Single GPU performance
- **Scaling Efficiency**: How well multiple GPUs are utilized
- **Resource Utilization**: GPU memory and compute usage

## Example Experiments

### Experiment 1: Config Size Comparison
```bash
# Run all config sizes for session v1
bash complete_training_pipeline.sh v1 small
bash complete_training_pipeline.sh v1 medium  
bash complete_training_pipeline.sh v1 large

# Analyze results
python analyze_training_results.py --session v1
```

### Experiment 2: Multiple Sessions
```bash
# Different sessions for A/B testing
bash complete_training_pipeline.sh baseline small
bash complete_training_pipeline.sh experiment_1 small
bash complete_training_pipeline.sh experiment_2 small

# Compare all sessions
python analyze_training_results.py
```

### Experiment 3: Real-time Monitoring
```bash
# Terminal 1: Start monitoring
watch -n 1 gpustat

# Terminal 2: Run training
bash complete_training_pipeline.sh monitoring_test medium

# Terminal 3: Watch logs
tail -f sessions/monitoring_test/single_gpu/trainer_state.json
```

## Troubleshooting

### Common Issues
1. **CUDA not detected**: Ensure CUDA drivers are installed
2. **Unsloth installation fails**: Continue without it, script will fallback
3. **Out of memory**: Use smaller batch sizes or smaller models
4. **Single GPU only**: Check accelerate configuration

### GPU Detection Issues
```bash
# Check CUDA
nvidia-smi

# Test PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reconfigure accelerate
accelerate config
```

### Performance Issues
- Use smaller configs (small/medium) for testing
- Reduce batch size in training_configs.py
- Check GPU memory with `nvidia-smi`

## Advanced Usage

### Custom Configurations
Edit `training_configs.py` to add custom model/dataset combinations.

### Custom Analysis
Modify `analyze_training_results.py` to add custom metrics or visualizations.

### Integration with Weights & Biases
Set `WANDB_DISABLED=false` in training scripts to enable W&B logging.

## Hardware Requirements

### Minimum
- 1 GPU with 8GB+ VRAM
- 16GB+ system RAM
- 50GB+ disk space

### Recommended  
- 2+ GPUs with 16GB+ VRAM each
- 32GB+ system RAM
- 100GB+ SSD storage

### For Large Config
- 2+ GPUs with 24GB+ VRAM each
- 64GB+ system RAM  
- 200GB+ storage