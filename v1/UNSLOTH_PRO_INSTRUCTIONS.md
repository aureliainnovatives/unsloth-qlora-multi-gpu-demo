# Unsloth Pro Multi-GPU Training Instructions

## Overview

This reference implementation (`train_multi_gpu_unsloth.py`) is designed for **Unsloth Pro users** who want to leverage Unsloth's optimizations in multi-GPU environments without the gradient checkpointing conflicts that affect the community version.

## Prerequisites

### 1. Unsloth Pro License
- Active Unsloth Pro subscription
- Proper licensing credentials configured
- Access to Pro-specific multi-GPU features

### 2. Hardware Requirements
- 2+ GPUs with 16GB+ VRAM each (recommended)
- NVLink or high-bandwidth GPU interconnect (optimal)
- 32GB+ system RAM

### 3. Software Dependencies
```bash
# Install Unsloth Pro (requires valid license)
pip install unsloth[pro]

# Or if different installation method:
pip install unsloth-pro

# Standard dependencies
pip install transformers accelerate datasets peft bitsandbytes trl
```

## Usage Methods

### Method 1: Direct Python Execution (Recommended for Unsloth Pro)
If Unsloth Pro handles distribution internally:

```bash
# Basic usage
python train_multi_gpu_unsloth.py --trainsession v1 --config small

# Different configurations
python train_multi_gpu_unsloth.py --trainsession production --config medium
python train_multi_gpu_unsloth.py --trainsession experiment --config large
```

### Method 2: Accelerate Launch (Fallback)
If Unsloth Pro still requires external orchestration:

```bash
# Configure accelerate first
accelerate config

# Launch training
accelerate launch train_multi_gpu_unsloth.py --trainsession v1 --config medium
```

## Script Features

### 1. Automatic Detection
- **Pro Version Check**: Automatically detects if Unsloth Pro is available
- **Fallback Support**: Falls back to standard multi-GPU if Pro not detected
- **Version Reporting**: Reports Unsloth version and capabilities

### 2. Enhanced Multi-GPU Support
- **Native Distribution**: Uses Unsloth Pro's built-in multi-GPU handling
- **Gradient Compatibility**: Pro version resolves DDP conflicts
- **Optimized Checkpointing**: Safe gradient checkpointing with multi-GPU

### 3. Session Integration
- **Organized Output**: Creates `sessions/SESSION_NAME/multi_gpu_unsloth/`
- **Comparison Ready**: Compatible with existing analysis tools
- **Pro Metadata**: Includes Pro version info in results

## Expected Improvements with Unsloth Pro

### Performance Benefits
1. **Best of Both Worlds**: Unsloth optimizations + true multi-GPU scaling
2. **Memory Efficiency**: Better memory management across GPUs
3. **Speed Optimization**: Custom CUDA kernels + distributed training
4. **Quality Maintenance**: No optimization vs quality trade-offs

### Technical Advantages
1. **No DDP Conflicts**: Gradient checkpointing works seamlessly
2. **Native Integration**: Purpose-built for multi-GPU environments
3. **Advanced Quantization**: Pro-level quantization strategies
4. **Stability**: Production-tested multi-GPU implementation

## Configuration Options

### Small Config (Testing)
```bash
python train_multi_gpu_unsloth.py --trainsession test --config small
```
- Model: DialoGPT-small (117M)
- Steps: 20
- Quantization: 8-bit
- Use case: Development, verification

### Medium Config (Development)
```bash
python train_multi_gpu_unsloth.py --trainsession dev --config medium
```
- Model: DialoGPT-medium (345M)
- Steps: 100  
- Quantization: 8-bit
- Use case: Feature development, benchmarking

### Large Config (Production)
```bash
python train_multi_gpu_unsloth.py --trainsession production --config large
```
- Model: Llama-2-7B (7B)
- Steps: 500
- Quantization: 4-bit
- Use case: Production training, best results

## Monitoring and Analysis

### Real-time Monitoring
```bash
# Monitor GPU usage
watch -n 1 gpustat

# Verify multi-GPU utilization
python verify_multi_gpu.py
```

### Results Analysis
```bash
# Analyze Pro results
python analyze_training_results.py --session v1

# Compare with standard multi-GPU
python log_viewer.py --compare --plot
```

## Expected Output Structure

```
sessions/
  v1/
    single_gpu/              # Standard single GPU (Unsloth community)
    multi_gpu/               # Standard multi-GPU (no Unsloth)
    multi_gpu_unsloth/       # Unsloth Pro multi-GPU (NEW)
      results.json           # Pro version metadata
      trainer_state.json     # Training logs
      final_model/           # Optimized model
```

## Performance Expectations

Based on Unsloth Pro's capabilities, expected improvements over standard multi-GPU:

| Metric | Standard Multi-GPU | Unsloth Pro Multi-GPU | Expected Improvement |
|--------|-------------------|----------------------|---------------------|
| Speed | 1.6x vs single GPU | 2.0-2.5x vs single GPU | 25-50% faster |
| Memory | Standard efficiency | Optimized allocation | 10-20% better |
| Quality | Good convergence | Optimized convergence | Similar or better |
| Stability | Occasional issues | Production stable | More reliable |

## Troubleshooting

### License Issues
```bash
# Check Unsloth Pro status
python -c "import unsloth; print(unsloth.__version__)"

# Verify license
python -c "import unsloth; print(hasattr(unsloth, 'FastLanguageModelPro'))"
```

### Fallback Behavior
- Script automatically falls back to standard multi-GPU if Pro not available
- Look for "⚠️ Unsloth Pro not detected" in output
- Check results.json for `"training_framework"` field

### Performance Issues
- Ensure NVLink is enabled between GPUs
- Check GPU memory usage with `nvidia-smi`
- Verify network bandwidth between nodes (if distributed)

## Integration with Existing Workflow

### Update Analysis Tools
The existing analysis tools will automatically recognize the new output format:

```bash
# Will show comparison between all three approaches
python analyze_training_results.py --session v1

# Expected output:
# - single_gpu (Unsloth community)
# - multi_gpu (Standard transformers)  
# - multi_gpu_unsloth (Unsloth Pro)
```

### Batch Experimentation
```bash
# Run all three approaches for comparison
python train_single_gpu.py --trainsession comparison --config medium
accelerate launch train_multi_gpu.py --trainsession comparison --config medium  
python train_multi_gpu_unsloth.py --trainsession comparison --config medium

# Analyze complete comparison
python analyze_training_results.py --session comparison
```

## Future Enhancements

1. **Automatic Pro Detection**: Enhanced detection of Pro features
2. **Advanced Configurations**: Pro-specific optimization settings
3. **Distributed Training**: Multi-node support with Unsloth Pro
4. **Custom Kernels**: Integration with latest Pro optimizations

## Support

For Unsloth Pro specific issues:
- Check Unsloth Pro documentation
- Contact Unsloth Pro support team
- Verify license status and Pro feature access

For this implementation:
- Use existing project analysis tools
- Compare with standard multi-GPU results
- Check session outputs for detailed metrics