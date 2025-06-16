# Multi-GPU vs Single GPU: Technical Coding Differences

## Overview

This document explains the specific coding differences that enable multi-GPU processing beyond just using Unsloth or PyTorch. Understanding these key technical components is crucial for implementing distributed training effectively.

## Core Multi-GPU Enablers

### 1. Accelerate Framework Integration

The most fundamental difference is the initialization of the distributed environment:

**Single GPU:**
```python
# No special initialization needed
model = FastLanguageModel.from_pretrained(model_name)
```

**Multi-GPU:**
```python
from accelerate import Accelerator

# This ONE line enables distributed training
accelerator = Accelerator()

# These properties give you multi-GPU context
accelerator.num_processes      # Total number of GPU processes
accelerator.local_process_index # Current GPU index (0, 1, 2, etc.)
accelerator.is_main_process    # True only for GPU 0 (coordinator)
```

### 2. Device Mapping Strategy

Device placement becomes critical in multi-GPU setups:

**Single GPU:**
```python
# Model loads on default device
model = AutoModelForCausalLM.from_pretrained(model_name)
```

**Multi-GPU:**
```python
# CRITICAL: Map model to specific GPU for each process
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": accelerator.local_process_index}  # Each process gets its GPU
)
```

**Why this matters:**
- Each GPU process must load the model on its assigned GPU
- `accelerator.local_process_index` ensures process 0 uses GPU 0, process 1 uses GPU 1, etc.
- Without this, all processes would try to use GPU 0

### 3. Process Coordination

Output and logging must be coordinated to avoid conflicts:

**Single GPU:**
```python
# All operations happen on one process
print("Training started")
trainer.train()
print("Training finished")
```

**Multi-GPU:**
```python
# Only main process prints to avoid spam
if accelerator.is_main_process:
    print("Training started")

trainer.train()  # All processes train in parallel

if accelerator.is_main_process:
    print("Training finished")  # Only main process reports
```

**Why this is necessary:**
- Without coordination, you'd see duplicate log messages (one per GPU)
- File saving operations should only happen once
- Progress reporting should be centralized

### 4. Launch Method Difference

The execution method fundamentally changes:

**Single GPU:**
```bash
# Direct execution
python train_single_gpu.py --config medium
```

**Multi-GPU:**
```bash
# Accelerate orchestrates multiple processes
accelerate launch train_multi_gpu.py --config medium
```

**What `accelerate launch` does:**
- Spawns N processes (one per GPU)
- Each process runs the same script but with different environment variables
- Sets up communication between processes
- Manages process lifecycle and cleanup

## Batch Size Scaling

Multi-GPU training automatically scales batch sizes:

**Single GPU:**
```python
# Total batch size = per_device_batch_size × gradient_accumulation_steps
total_batch = 2 × 4 = 8
```

**Multi-GPU:**
```python
# Total batch size = per_device_batch_size × gradient_accumulation_steps × num_gpus
total_batch = 2 × 4 × 2 = 16  # Automatically doubles with 2 GPUs
```

This scaling happens automatically - you don't need to modify your batch size settings.

## Environment Variables: The Hidden Orchestration

When you run `accelerate launch`, it sets critical environment variables:

```python
import os

# Multi-GPU process coordination
RANK = os.environ.get('RANK')           # Global process rank (0, 1, 2, ...)
LOCAL_RANK = os.environ.get('LOCAL_RANK') # Local GPU index (0, 1)
WORLD_SIZE = os.environ.get('WORLD_SIZE') # Total number of processes

# PyTorch distributed backend communication
MASTER_ADDR = os.environ.get('MASTER_ADDR', 'localhost')  # Coordinator address
MASTER_PORT = os.environ.get('MASTER_PORT', '12355')      # Communication port
```

These variables enable:
- Process identification and coordination
- Network communication setup
- Gradient synchronization
- Distributed data loading

## Gradient Synchronization: The Core Magic

Understanding how gradients are handled differently:

**Single GPU:**
```python
# Gradients computed and applied immediately
loss.backward()
optimizer.step()
```

**Multi-GPU (under the hood):**
```python
# Each GPU computes gradients on its batch
loss.backward()  

# PyTorch DDP automatically:
# 1. Collects gradients from all GPUs
# 2. Averages them across processes  
# 3. Broadcasts averaged gradients back to all GPUs
# 4. All GPUs apply the same averaged gradients

optimizer.step()  # Synchronized step across all GPUs
```

**The DDP Process:**
1. **Forward Pass**: Each GPU processes different data batches
2. **Backward Pass**: Each GPU computes gradients for its batch
3. **AllReduce**: Gradients are averaged across all GPUs
4. **Update**: All GPUs apply the same averaged gradients
5. **Synchronization**: Model parameters stay identical across GPUs

## File I/O and Output Handling

Output operations require careful coordination:

**Single GPU:**
```python
# Direct output path
output_dir = "./single_gpu_output"
trainer.save_model(output_dir)
```

**Multi-GPU:**
```python
# Only main process saves to avoid conflicts
if accelerator.is_main_process:
    trainer.save_model("./multi_gpu_output")
    
    # Save results and metrics
    with open("results.json", "w") as f:
        json.dump(results, f)

# All processes can write to different files if needed
with open(f"debug_gpu_{accelerator.local_process_index}.log", "w") as f:
    f.write("Process-specific debugging info")
```

## Training Arguments for Multi-GPU Optimization

Specific parameters that optimize multi-GPU performance:

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    # Multi-GPU specific optimizations
    ddp_find_unused_parameters=False,    # Optimize DDP communication
    dataloader_pin_memory=False,         # Avoid memory conflicts  
    dataloader_num_workers=0,            # Prevent worker conflicts
    
    # Gradient and memory handling
    gradient_checkpointing=False,        # May conflict with DDP
    fp16=True,                          # Mixed precision for speed
    bf16=torch.cuda.is_bf16_supported(), # Use bf16 if available
    
    # Communication optimization
    ddp_bucket_cap_mb=25,               # Control gradient bucket size
    ddp_broadcast_buffers=False,        # Reduce communication overhead
    
    # Standard settings that work well with multi-GPU
    per_device_train_batch_size=2,      # Per-GPU batch size
    gradient_accumulation_steps=4,      # Accumulate before sync
    save_total_limit=1,                 # Limit checkpoint storage
    report_to=None,                     # Disable external logging
)
```

## Complete Minimal Multi-GPU Example

Here's the absolute minimum code difference:

**Single GPU (4 lines):**
```python
from transformers import Trainer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("model_name")
trainer = Trainer(model=model, args=args, train_dataset=dataset)
trainer.train()
```

**Multi-GPU (6 lines + different launch):**
```python
from transformers import Trainer, AutoModelForCausalLM
from accelerate import Accelerator

accelerator = Accelerator()  # +1 line: Initialize distributed environment
model = AutoModelForCausalLM.from_pretrained(
    "model_name", 
    device_map={"": accelerator.local_process_index}  # +1 parameter: GPU mapping
)
trainer = Trainer(model=model, args=args, train_dataset=dataset)
trainer.train()
```

**Launch Commands:**
```bash
# Single GPU - direct execution
python train_script.py --config medium

# Multi-GPU - distributed execution
accelerate launch train_script.py --config medium
```

## Data Loading Considerations

Dataset handling requires attention in multi-GPU setups:

**Automatic Distribution:**
```python
# Accelerate automatically handles data distribution
dataset = load_dataset("dataset_name", split="train")

# Each GPU gets different batches automatically
# No explicit data splitting needed
```

**Manual Control (if needed):**
```python
# Advanced: Manual data distribution
if accelerator.is_main_process:
    dataset = load_dataset("dataset_name", split="train")
else:
    dataset = None

# Broadcast dataset to all processes
dataset = accelerator.prepare(dataset)
```

## Debugging Multi-GPU Issues

Common debugging patterns:

**Process Identification:**
```python
print(f"Process {accelerator.local_process_index}/{accelerator.num_processes}")
print(f"Main process: {accelerator.is_main_process}")
print(f"Device: {accelerator.device}")
```

**GPU Memory Monitoring:**
```python
if accelerator.is_main_process:
    import torch
    for i in range(torch.cuda.device_count()):
        memory_used = torch.cuda.memory_allocated(i) / 1024**3
        memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"GPU {i}: {memory_used:.1f}GB / {memory_total:.1f}GB")
```

## Performance Considerations

### Communication Overhead
- **Small models**: Communication overhead may reduce efficiency
- **Large models**: Better scaling due to computation/communication ratio
- **Network bandwidth**: Faster interconnects (NVLink) improve scaling

### Memory Distribution
```python
# Memory per GPU is roughly:
memory_per_gpu = total_model_memory / num_gpus + batch_memory_per_gpu

# Batch memory scales with:
batch_memory = per_device_batch_size * sequence_length * model_hidden_size
```

### Scaling Efficiency
Expected scaling with different model sizes:
- **Small models (< 1B params)**: 60-80% efficiency
- **Medium models (1-10B params)**: 80-90% efficiency  
- **Large models (> 10B params)**: 90-95% efficiency

## The Three Key Enablers Summary

1. **`accelerator = Accelerator()`** - Sets up distributed environment and process coordination
2. **`device_map={"": accelerator.local_process_index}`** - Maps model to specific GPU for each process
3. **`accelerate launch`** - Spawns multiple processes with proper environment variables and communication setup

## Advanced Multi-GPU Patterns

### Conditional Multi-GPU Logic
```python
from accelerate import Accelerator

accelerator = Accelerator()

if accelerator.num_processes > 1:
    # Multi-GPU specific optimizations
    training_args.ddp_find_unused_parameters = False
    training_args.dataloader_num_workers = 0
else:
    # Single GPU optimizations
    training_args.dataloader_num_workers = 4
    training_args.dataloader_pin_memory = True
```

### Mixed Precision Coordination
```python
# Automatic mixed precision with multi-GPU
training_args = TrainingArguments(
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    # Accelerate handles mixed precision across GPUs automatically
)
```

## Conclusion

The transition from single GPU to multi-GPU training requires minimal code changes but significant understanding of distributed computing concepts. The key insight is that most complexity is handled automatically by the Accelerate framework and PyTorch's DDP, but developers must understand:

1. **Process coordination** - Multiple processes running the same code
2. **Device mapping** - Ensuring each process uses the correct GPU
3. **Communication patterns** - How gradients and data are synchronized
4. **Launch methodology** - Using distributed launchers instead of direct execution

With these fundamentals, multi-GPU training becomes a straightforward extension of single GPU training, providing significant performance improvements for larger models and datasets.