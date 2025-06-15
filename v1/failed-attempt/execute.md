# 🌅 EXECUTE GUIDE: Multi-GPU QLoRA Training System

**"Just Woke Up" Complete Guide** - Everything you need to run, monitor, and prove single vs multi-GPU training.

## 🎯 **What We Built (Quick Recap)**

A complete **QLoRA Multi-GPU training system** for **Qwen-7B** with:
- ✅ **Configurable GPU modes** (single/multi)
- ✅ **Real-time monitoring** and logging
- ✅ **Performance comparison tools**
- ✅ **Evidence collection** for demonstrations

## 🚀 **STEP-BY-STEP EXECUTION**

### **STEP 1: Environment Check**
```bash
# Check if everything is installed
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"
nvidia-smi
```

Expected output:
```
PyTorch: 2.0.0+cu121
GPUs available: 2
```

### **STEP 2: Auto-Setup Accelerate** ⭐
```bash
# Run our automated setup script
./setup_accelerate.sh
```

This script will:
- ✅ Detect your GPUs automatically
- ✅ Create optimal accelerate config
- ✅ Verify setup
- ✅ Show next steps

**Alternative Manual Setup:**
```bash
accelerate config
# Choose: Multi-GPU, bf16, no DeepSpeed, yes torch.compile
```

### **STEP 3: Single GPU Demo (Baseline)**

#### **3A. Start Single GPU Training**
```bash
# Terminal 1: Training
python train.py --force-single-gpu
```

#### **3B. Monitor Single GPU (New Terminal)**
```bash
# Terminal 2: Real-time monitoring
python monitor_gpu.py --output single_gpu_demo.json

# OR system monitoring
watch -n 1 nvidia-smi
```

#### **3C. Watch Training Logs (New Terminal)**
```bash
# Terminal 3: Clean training progress
tail -f logs/training_*/training_*_main.log
```

### **STEP 4: Multi-GPU Demo (Comparison)**

#### **4A. Start Multi-GPU Training**
```bash
# Terminal 1: Training
accelerate launch train.py
```

#### **4B. Monitor Multi-GPU (New Terminal)**
```bash
# Terminal 2: Real-time monitoring
python monitor_gpu.py --output multi_gpu_demo.json

# OR system monitoring
watch -n 1 nvidia-smi
```

#### **4C. Watch Multi-GPU Operations (New Terminal)**
```bash
# Terminal 3: Multi-GPU coordination logs
tail -f logs/training_*/training_*_multigpu.log
```

### **STEP 5: Analyze and Compare Results**
```bash
# Compare all training sessions
python analyze_logs.py --compare

# Generate detailed report
python analyze_logs.py --report

# Create performance plots
python analyze_logs.py --plot

# Analyze multi-GPU communication
python analyze_logs.py --communication
```

## 📊 **REAL-TIME MONITORING OPTIONS**

### **Option 1: Our Custom Monitor** (Recommended)
```bash
# Continuous monitoring with JSON output
python monitor_gpu.py --output training_monitor.json

# Single snapshot
python monitor_gpu.py --snapshot

# Monitor during training automatically
python monitor_gpu.py --monitor-training 'accelerate launch train.py'
```

### **Option 2: System Tools**
```bash
# Basic GPU monitoring
nvidia-smi

# Continuous system monitoring
watch -n 1 nvidia-smi

# Detailed GPU info
nvidia-smi -l 1  # Update every 1 second

# GPU memory details
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1
```

### **Option 3: Advanced Monitoring**
```bash
# GPU utilization and memory
nvidia-smi dmon -s u

# Process monitoring
nvidia-smi pmon

# Full system monitoring with htop
htop
```

### **Option 4: Web-based Monitoring** (Optional)
```bash
# Install nvidia-ml-py for web dashboard
pip install nvidia-ml-py3

# Our monitoring provides JSON data you can visualize
python analyze_logs.py --plot  # Creates graphs
```

## 🔍 **EVIDENCE COLLECTION**

### **✅ Proof of Single GPU Usage:**

**What you'll see:**
```bash
# In terminal output:
============================================================
GPU CONFIGURATION
============================================================
Mode: SINGLE-GPU
Available GPUs: 2
CUDA_VISIBLE_DEVICES: 0
============================================================

# In logs:
FORCED SINGLE GPU MODE - Using GPU 0 only

# In nvidia-smi:
Only GPU 0 shows high utilization
```

### **✅ Proof of Multi-GPU Usage:**

**What you'll see:**
```bash
# In terminal output:
============================================================
GPU CONFIGURATION
============================================================
Mode: MULTI-GPU
Available GPUs: 2
CUDA_VISIBLE_DEVICES: Not set
============================================================

# In logs:
📡 MULTI-GPU CONFIGURATION:
  • Accelerator processes: 2
  • Distributed backend: nccl
  • World size: 2

Communication | gradient_sync | Rank 0 | Size: 45.67MB

# In nvidia-smi:
Both GPUs show high utilization
```

### **✅ Performance Comparison:**
```bash
# Analysis output will show:
PERFORMANCE COMPARISON:
Single GPU: 0.83 steps/sec, 85.2% GPU util
Multi-GPU: 1.54 steps/sec, 82.0% GPU util
🚀 Multi-GPU Speedup: 1.85x
📊 GPU Efficiency: 92.5%
```

## 🎮 **QUICK DEMO COMMANDS**

### **One-Command Demos:**
```bash
# Quick single GPU test (runs for 50 steps)
python train.py --force-single-gpu --max-steps 50

# Quick multi-GPU test (runs for 50 steps)
accelerate launch train.py --max-steps 50

# Monitor any training session
python monitor_gpu.py --monitor-training 'your_training_command_here'
```

### **Evidence Collection Commands:**
```bash
# Collect single GPU evidence
python train.py --force-single-gpu & python monitor_gpu.py --output single_evidence.json

# Collect multi-GPU evidence  
accelerate launch train.py & python monitor_gpu.py --output multi_evidence.json

# Compare evidence
python analyze_logs.py --compare
```

## 🔧 **CONFIGURATION QUICK CHANGES**

### **Change Training Length (in config.py):**
```python
max_steps = 100  # Quick demo (default: 1000)
```

### **Change Batch Size:**
```python
per_device_train_batch_size = 1  # Smaller for less memory
```

### **Force Specific GPUs:**
```bash
python train.py --gpu-ids 0,1  # Use specific GPUs
python train.py --gpu-ids 0    # Use only GPU 0
```

### **Change Model/Dataset (in config.py):**
```python
model_name = "unsloth/qwen-7b-qlora"  # Change model
dataset_name = "timdettmers/openassistant-guanaco"  # Change dataset
```

## 🚨 **TROUBLESHOOTING**

### **Issue: "No GPUs found"**
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### **Issue: "Multi-GPU not working"**
```bash
# Check accelerate config
accelerate env

# Re-run setup
./setup_accelerate.sh

# Verify distributed training
python -c "import torch.distributed as dist; print('Distributed available:', dist.is_available())"
```

### **Issue: "Out of memory"**
**Reduce batch size in config.py:**
```python
per_device_train_batch_size = 1  # Reduce from 2
gradient_accumulation_steps = 8  # Increase to maintain effective batch size
```

### **Issue: "Training too slow"**
**Check:**
```bash
# GPU utilization
nvidia-smi

# Training logs for bottlenecks
tail -f logs/training_*/training_*_performance.log
```

## 📋 **QUICK REFERENCE**

### **Essential Commands:**
```bash
# Setup (run once)
./setup_accelerate.sh

# Single GPU training
python train.py --force-single-gpu

# Multi-GPU training  
accelerate launch train.py

# Monitor training
python monitor_gpu.py

# Compare results
python analyze_logs.py --compare

# Emergency stop
Ctrl+C
```

### **File Locations:**
- **Training logs**: `logs/training_YYYYMMDD_HHMMSS/`
- **Monitoring data**: `*.json` files
- **Config**: `config.py`
- **Main script**: `train.py`

## 🎯 **SUCCESS INDICATORS**

### **Single GPU Success:**
- ✅ Logs show "SINGLE GPU MODE"
- ✅ Only GPU 0 active in nvidia-smi
- ✅ Training progresses normally

### **Multi-GPU Success:**
- ✅ Logs show "MULTI-GPU MODE"
- ✅ All GPUs active in nvidia-smi
- ✅ Higher training speed than single GPU
- ✅ Communication logs show gradient sync

### **System Working Properly:**
- ✅ No CUDA errors in logs
- ✅ GPU utilization >70%
- ✅ Training loss decreasing
- ✅ Analysis shows speedup >1.5x

---

**🎮 TL;DR**: Run `./setup_accelerate.sh`, then `accelerate launch train.py` for multi-GPU or `python train.py --force-single-gpu` for single GPU. Monitor with `nvidia-smi` and analyze with `python analyze_logs.py --compare`!

**Ready to demonstrate multi-GPU power? 🚀**