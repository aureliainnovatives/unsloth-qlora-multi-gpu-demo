# 📊 Enhanced Multi-GPU Training Logging Guide

This guide explains the comprehensive logging system that provides clear insights into multi-GPU training distribution and networking.

## 🗂️ Log File Structure

When training starts, a unique session directory is created under `./logs/`:

```
logs/
└── training_20241206_143022/           # Session ID with timestamp
    ├── training_20241206_143022_main.log         # Main readable training log
    ├── training_20241206_143022_multigpu.log     # Multi-GPU specific operations
    ├── training_20241206_143022_performance.log  # Performance monitoring
    ├── training_20241206_143022_summary.json     # Comprehensive session data
    └── training_20241206_143022_summary.txt      # Human-readable summary
```

## 📋 Log File Contents

### 1. Main Training Log (`*_main.log`)
**Readable, concise training progress:**
```
14:30:22 | INFO  | 🚀 TRAINING SESSION STARTED
14:30:22 | INFO  | Session ID: training_20241206_143022
14:30:22 | INFO  | Mode: MULTI-GPU
14:30:22 | INFO  | GPU Count: 2
14:30:22 | INFO  | Model: unsloth/qwen-7b-qlora

📡 MULTI-GPU CONFIGURATION:
  • Accelerator processes: 2
  • Current process rank: 0
  • Mixed precision: bf16
  • GPU 0: NVIDIA RTX 4090 (24.0GB)
  • GPU 1: NVIDIA RTX 4090 (24.0GB)

Step   50/1000 ( 5.0%) | Loss: 2.1234 | LR: 1.50e-04 | Speed: 2.5 steps/s | ETA: 6.3min
  └─ Metrics: step_duration: 0.400, samples_per_second: 5.0

📊 EVALUATION RESULTS:
  • eval_loss: 1.8765
  • eval_perplexity: 6.5432
```

### 2. Multi-GPU Log (`*_multigpu.log`)
**Multi-GPU coordination and communication:**
```
2024-12-06 14:30:25 | INFO     | MainProcess  | Distributed backend: nccl
2024-12-06 14:30:25 | INFO     | MainProcess  | World size: 2
2024-12-06 14:30:25 | INFO     | MainProcess  | Local rank: 0

2024-12-06 14:30:45 | INFO     | MainProcess  | Multi-GPU Step 20 | Processes: 2 | Local rank: 0 | Process index: 0
2024-12-06 14:30:45 | INFO     | MainProcess  | Communication | gradient_sync | Rank 0 | Size: 45.67MB | Duration: 12.34ms | Bandwidth: 3701.46MB/s
```

### 3. Performance Log (`*_performance.log`)
**Performance metrics and bottleneck detection:**
```
2024-12-06 14:32:15 | INFO     | MainProcess  | Performance | Step 100 | Avg step time: 0.380s | Throughput: 10.5 samples/s | Total batch size: 4
2024-12-06 14:32:15 | WARNING  | MainProcess  | GPU 1 utilization low: 45.2%
```

## 🎯 Key Multi-GPU Indicators in Logs

### ✅ Successful Multi-GPU Training Signs:
1. **Session Start:**
   ```
   Mode: MULTI-GPU
   GPU Count: 2 (or more)
   Accelerator processes: 2
   ```

2. **Communication Logs:**
   ```
   Communication | gradient_sync | Size: 45.67MB | Bandwidth: 3701MB/s
   Multi-GPU Step 20 | Processes: 2 | Local rank: 0
   ```

3. **Performance Metrics:**
   ```
   Total batch size: 8  # Higher than single GPU
   Throughput: 15.2 samples/s  # Higher than single GPU
   ```

### ❌ Single GPU Training Signs:
1. **Session Start:**
   ```
   Mode: SINGLE-GPU
   GPU Count: 1
   FORCED SINGLE GPU MODE - Using GPU 0 only
   ```

2. **No Communication Logs:**
   - No gradient_sync operations
   - No multi-GPU coordination messages

## 🔍 Monitoring Commands

### Real-time Monitoring:
```bash
# Watch main training log
tail -f logs/training_*/training_*_main.log

# Monitor multi-GPU operations
tail -f logs/training_*/training_*_multigpu.log

# Watch for performance issues
tail -f logs/training_*/training_*_performance.log
```

### GPU Usage Monitoring:
```bash
# System GPU monitoring
watch -n 1 nvidia-smi

# Our custom monitoring
python monitor_gpu.py --output current_training.json

# Monitor during training
python monitor_gpu.py --monitor-training 'accelerate launch train.py'
```

## 📈 Log Analysis Tools

### 1. Compare Training Sessions:
```bash
# Compare all sessions
python analyze_logs.py --compare

# Analyze specific session
python analyze_logs.py --session-id training_20241206_143022

# Generate comprehensive report
python analyze_logs.py --report

# Create training plots
python analyze_logs.py --plot
```

### 2. Multi-GPU Communication Analysis:
```bash
python analyze_logs.py --communication
```

**Sample Output:**
```
📡 Session: training_20241206_143022
  GPUs: 2
  Communication Operations:
    gradient_sync: 45 occurrences
      Avg data size: 67.23MB
      Total data: 3025.35MB
    training_start: 1 occurrences
```

## 🎮 Demo Commands for Evidence Collection

### Single GPU Demo:
```bash
# Terminal 1: Start training with single GPU
python train.py --force-single-gpu

# Terminal 2: Monitor in real-time
tail -f logs/training_*/training_*_main.log
```

### Multi-GPU Demo:
```bash
# Terminal 1: Start multi-GPU training
accelerate launch train.py

# Terminal 2: Monitor multi-GPU operations
tail -f logs/training_*/training_*_multigpu.log

# Terminal 3: Monitor system GPUs
watch -n 1 nvidia-smi
```

## 📊 Log Analysis Examples

### Performance Comparison:
```bash
python analyze_logs.py --compare
```

**Sample Output:**
```
SINGLE GPU vs MULTI-GPU COMPARISON
============================================================

📱 SINGLE GPU SESSIONS (1):
  Session: training_20241206_140000
  Duration: 1200.5s (20.0min)
  GPUs: 1
  Steps: 1000
  Final Loss: 1.2345
  Steps/sec: 0.83
  GPU 0: 85.2% avg util, 98.1% peak, 78.3% avg memory

🔗 MULTI-GPU SESSIONS (1):
  Session: training_20241206_143022  
  Duration: 650.2s (10.8min)
  GPUs: 2
  Steps: 1000
  Final Loss: 1.2340
  Steps/sec: 1.54
  GPU 0: 82.1% avg util, 95.2% peak, 75.1% avg memory
  GPU 1: 81.8% avg util, 94.8% peak, 74.9% avg memory

⚡ PERFORMANCE COMPARISON:
Single GPU Average:
  Steps/sec: 0.83
  GPU utilization: 85.2%

Multi-GPU Average:
  Steps/sec: 1.54
  GPU utilization: 82.0%

🚀 Multi-GPU Speedup: 1.85x
📊 GPU Efficiency: 92.5%
```

## 🏗️ Log Configuration

### Customize Logging in `config.py`:
```python
# Enhanced logging settings
log_interval_steps: int = 50      # How often to log training steps
performance_log_steps: int = 100  # Performance monitoring frequency
gpu_snapshot_interval: int = 10   # GPU monitoring interval (seconds)
detailed_communication_logs: bool = True  # Log GPU communication
```

### Environment Variables:
```bash
# Control log verbosity
export TRAINING_LOG_LEVEL=INFO
export MULTIGPU_LOG_LEVEL=DEBUG

# Enable detailed NCCL logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

## 🔧 Troubleshooting with Logs

### Issue: Multi-GPU not working
**Check logs for:**
```
Mode: SINGLE-GPU  # Should be MULTI-GPU
FORCED SINGLE GPU MODE  # Remove force flags
CUDA_VISIBLE_DEVICES=0  # Should show multiple GPUs
```

### Issue: Poor GPU utilization
**Check performance log for:**
```
WARNING | GPU 1 utilization low: 35.2%
WARNING | Slow training detected: 15.1s per step
```

### Issue: Communication problems
**Check multi-GPU log for:**
```
ERROR | Distributed backend initialization failed
WARNING | NCCL communication timeout
```

This comprehensive logging system provides complete visibility into your multi-GPU training process with clear evidence of proper distribution and networking!