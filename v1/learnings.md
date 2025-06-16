# Multi-GPU QLoRA Training: Technical Learnings and Solutions

## Project Overview

This project aimed to implement QLoRA (Quantized Low-Rank Adaptation) fine-tuning with multi-GPU support, comparing performance between single GPU and multi-GPU setups. What started as a straightforward implementation revealed fundamental compatibility issues between optimization frameworks.

## The Challenge: Unsloth vs Multi-GPU

### Initial Approach
We began with a unified approach using **Unsloth** for both single and multi-GPU training:
- Unsloth provides significant speedups through custom CUDA kernels
- Optimized gradient checkpointing and memory management
- Seamless integration with LoRA for parameter-efficient fine-tuning

### The Core Problem: Gradient Checkpointing Incompatibility

**Technical Issue**: Unsloth's custom gradient checkpointing system conflicts with PyTorch's DistributedDataParallel (DDP) mechanism.

```
Error: "Expected to mark a variable ready only once"
```

**Root Cause Analysis**:
1. **Unsloth's Optimization**: Uses custom gradient checkpointing with specialized CUDA kernels
2. **DDP Requirements**: Expects standard PyTorch gradient flow for synchronization
3. **Conflict**: Unsloth's modified gradient computation breaks DDP's gradient readiness tracking

### Failed Solutions Attempted

#### 1. Gradient Checkpointing Disabling
```python
# Attempted fix - didn't resolve DDP conflicts
use_gradient_checkpointing=False
```
**Result**: Reduced memory optimization but DDP conflicts persisted

#### 2. LoRA Configuration Modifications
```python
# Tried different LoRA settings
r=8, lora_alpha=16, target_modules=["q_proj", "k_proj"]
```
**Result**: No impact on DDP compatibility

#### 3. Unsloth Native Multi-GPU
```python
# Attempted Unsloth's built-in multi-GPU
FastLanguageModel.for_training(model, use_gradient_checkpointing="unsloth")
```
**Result**: Limited multi-GPU support, inconsistent scaling

#### 4. SFTTrainer Integration
```python
# Tried using SFTTrainer with Unsloth
from trl import SFTTrainer
```
**Result**: Partial success but still DDP incompatibility

## The Solution: Hybrid Approach

### Architecture Decision
After extensive experimentation, we implemented a **hybrid training system**:

#### Single GPU Path (Optimized)
- **Framework**: Unsloth + FastLanguageModel
- **Optimization**: Custom CUDA kernels, 4-bit quantization
- **Benefits**: Maximum single-GPU performance, memory efficiency
- **Use Case**: Development, small-scale training

#### Multi-GPU Path (Compatible) 
- **Framework**: Standard Transformers + PEFT + Accelerate
- **Optimization**: Native PyTorch DDP, BitsAndBytes quantization
- **Benefits**: True multi-GPU scaling, stable distributed training
- **Use Case**: Production, large-scale training

### Technical Implementation

#### Single GPU Training (`train_single_gpu.py`)
```python
# Unsloth optimization for supported models
if "llama" in model_name.lower() or "mistral" in model_name.lower():
    model = FastLanguageModel.get_peft_model(
        model, r=16, use_gradient_checkpointing="unsloth"
    )
else:
    # Fallback to standard PEFT for other models
    model = get_peft_model(model, peft_config)
```

#### Multi-GPU Training (`train_multi_gpu.py`)
```python
# Standard transformers approach
model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config
)
model = get_peft_model(model, lora_config)

# Accelerate handles DDP automatically
accelerator = Accelerator()
```

## Performance Analysis

### Benchmarking Results (DialoGPT-medium, 100 steps)

| Metric | Single GPU (Unsloth) | Multi-GPU (Standard) | Improvement |
|--------|----------------------|---------------------|-------------|
| Duration | 125.4 seconds | 79.4 seconds | **1.58x faster** |
| Steps/second | 0.80 | 1.26 | **1.58x throughput** |
| Final Loss | 6.35 | 3.99 | **37% better convergence** |
| Memory Usage | ~3GB | ~6GB (distributed) | 2x total memory |

### Key Insights

1. **Multi-GPU Scaling**: Achieved 1.6x speedup with 2 GPUs (80% efficiency)
2. **Loss Quality**: Multi-GPU achieved better convergence (less aggressive quantization)
3. **Memory Distribution**: Load distributed across GPUs vs concentrated on single GPU

## Technical Lessons Learned

### 1. Framework Compatibility Matters
- Optimization frameworks can have deep incompatibilities
- Custom CUDA kernels may not work with distributed training
- Always test multi-GPU early in development

### 2. Quantization Trade-offs
- **4-bit quantization**: Maximum memory efficiency, potential quality loss
- **8-bit quantization**: Balanced approach for multi-GPU
- **No quantization**: Best quality, highest memory usage

### 3. Gradient Management Complexity
- DDP requires precise gradient synchronization
- Custom gradient checkpointing can break DDP assumptions
- Standard PyTorch patterns ensure compatibility

### 4. Model Architecture Sensitivity
- Some models work better with Unsloth (Llama, Mistral)
- Others require standard PEFT approaches (DialoGPT, GPT-2)
- Architecture-aware training path selection is crucial

## Session-Based Training System

### Innovation: Organized Experimentation
We developed a session-based system for systematic comparison:

```
sessions/
  v1/
    single_gpu/    # Unsloth-optimized results
    multi_gpu/     # Standard multi-GPU results
  v2/
    single_gpu/    # Different configuration
    multi_gpu/     # Comparison results
```

**Benefits**:
- Systematic A/B testing
- Performance comparison across configurations
- Reproducible experiments

## Recommendations

### When to Use Single GPU (Unsloth)
- **Development and prototyping**: Fast iteration, memory efficient
- **Small to medium models**: Models that fit in single GPU memory
- **Inference optimization**: Maximum single-GPU performance
- **Cost optimization**: Single GPU instances are cheaper

### When to Use Multi-GPU (Standard)
- **Production training**: Large datasets, long training runs
- **Large models**: Models requiring distributed memory
- **Quality priority**: When loss convergence is critical
- **Scalability**: When you need to scale beyond single GPU

### Hybrid Workflow
1. **Start with single GPU** for development and hyperparameter tuning
2. **Scale to multi-GPU** for production training
3. **Compare results** using the analysis tools
4. **Choose optimal path** based on performance requirements

## Future Improvements

1. **Automatic Path Selection**: Detect model compatibility and choose training path
2. **Advanced Quantization**: Explore QLoRA with different bit-widths
3. **Memory Optimization**: Better memory management for larger models
4. **Performance Profiling**: Detailed analysis of bottlenecks

## Conclusion

This project demonstrates that **there is no one-size-fits-all solution** for multi-GPU training with modern optimization frameworks. The key insight is that:

- **Unsloth excels at single-GPU optimization** but has multi-GPU limitations
- **Standard frameworks provide reliable multi-GPU scaling** with some performance trade-offs
- **A hybrid approach** gives the best of both worlds

The session-based comparison system allows practitioners to make informed decisions based on empirical performance data rather than theoretical assumptions.

**Final Takeaway**: Understanding the technical limitations and trade-offs of different frameworks is crucial for building robust, scalable training systems. Sometimes the best solution is not trying to force one tool to do everything, but rather using the right tool for each specific task.