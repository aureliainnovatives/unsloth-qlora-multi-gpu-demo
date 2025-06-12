#!/usr/bin/env python3

import os
import logging
import json
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import torch
import torch.distributed as dist
from accelerate import Accelerator
import psutil
import GPUtil


@dataclass
class TrainingSession:
    """Training session metadata"""
    session_id: str
    start_time: datetime
    mode: str  # "single-gpu" or "multi-gpu"
    gpu_count: int
    model_name: str
    dataset_name: str
    batch_size: int
    learning_rate: float
    max_steps: int


class MultiGPUTrainingLogger:
    """Enhanced logging system for multi-GPU training insights"""
    
    def __init__(self, config, accelerator: Accelerator, session_id: str = None):
        self.config = config
        self.accelerator = accelerator
        self.session_id = session_id or f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create logs directory structure
        self.log_dir = os.path.join("./logs", self.session_id)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup different log files
        self.setup_loggers()
        
        # Training metrics
        self.training_metrics = []
        self.gpu_snapshots = []
        self.communication_logs = []
        self.performance_metrics = []
        
        # Session info
        self.session = TrainingSession(
            session_id=self.session_id,
            start_time=datetime.now(),
            mode="multi-gpu" if accelerator.num_processes > 1 else "single-gpu",
            gpu_count=accelerator.num_processes,
            model_name=config.model_name,
            dataset_name=config.dataset_name,
            batch_size=config.per_device_train_batch_size,
            learning_rate=config.learning_rate,
            max_steps=config.max_steps
        )
        
        # Start background monitoring
        self.monitoring = True
        self.monitor_thread = None
        self.start_background_monitoring()
        
        # Log session start
        self.log_session_start()
    
    def setup_loggers(self):
        """Setup different specialized loggers"""
        
        # Main training logger - readable format
        self.main_logger = logging.getLogger(f"training_{self.session_id}")
        self.main_logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.main_logger.handlers.clear()
        
        # Console handler with clean format
        console_handler = logging.StreamHandler()
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)-5s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        console_handler.setLevel(logging.INFO)
        
        # File handler with detailed format
        file_handler = logging.FileHandler(
            os.path.join(self.log_dir, f"{self.session_id}_main.log")
        )
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(processName)-12s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        file_handler.setLevel(logging.DEBUG)
        
        # Only main process logs to console to avoid duplication
        if self.accelerator.is_main_process:
            self.main_logger.addHandler(console_handler)
        
        self.main_logger.addHandler(file_handler)
        
        # Multi-GPU specific logger
        self.multigpu_logger = logging.getLogger(f"multigpu_{self.session_id}")
        self.multigpu_logger.setLevel(logging.INFO)
        multigpu_handler = logging.FileHandler(
            os.path.join(self.log_dir, f"{self.session_id}_multigpu.log")
        )
        multigpu_handler.setFormatter(file_format)
        self.multigpu_logger.addHandler(multigpu_handler)
        
        # Performance logger
        self.perf_logger = logging.getLogger(f"performance_{self.session_id}")
        self.perf_logger.setLevel(logging.INFO)
        perf_handler = logging.FileHandler(
            os.path.join(self.log_dir, f"{self.session_id}_performance.log")
        )
        perf_handler.setFormatter(file_format)
        self.perf_logger.addHandler(perf_handler)
    
    def log_session_start(self):
        """Log training session start information"""
        
        self.main_logger.info("="*80)
        self.main_logger.info("🚀 TRAINING SESSION STARTED")
        self.main_logger.info("="*80)
        self.main_logger.info(f"Session ID: {self.session_id}")
        self.main_logger.info(f"Mode: {self.session.mode.upper()}")
        self.main_logger.info(f"GPU Count: {self.session.gpu_count}")
        self.main_logger.info(f"Model: {self.session.model_name}")
        self.main_logger.info(f"Dataset: {self.session.dataset_name}")
        self.main_logger.info(f"Batch Size (per device): {self.session.batch_size}")
        self.main_logger.info(f"Learning Rate: {self.session.learning_rate}")
        self.main_logger.info(f"Max Steps: {self.session.max_steps}")
        
        # Multi-GPU specific information
        if self.session.mode == "multi-gpu":
            self.log_multigpu_setup()
        
        self.main_logger.info("="*80)
    
    def log_multigpu_setup(self):
        """Log multi-GPU setup details"""
        
        self.main_logger.info("\n📡 MULTI-GPU CONFIGURATION:")
        self.main_logger.info(f"  • Accelerator processes: {self.accelerator.num_processes}")
        self.main_logger.info(f"  • Current process rank: {self.accelerator.local_process_index}")
        self.main_logger.info(f"  • Mixed precision: {self.accelerator.mixed_precision}")
        self.main_logger.info(f"  • Gradient accumulation: {self.config.gradient_accumulation_steps}")
        
        # Log GPU details
        if torch.cuda.is_available():
            self.main_logger.info(f"  • CUDA devices visible: {os.environ.get('CUDA_VISIBLE_DEVICES', 'All')}")
            
            for i in range(torch.cuda.device_count()):
                gpu_props = torch.cuda.get_device_properties(i)
                memory_gb = gpu_props.total_memory / 1024**3
                self.main_logger.info(f"  • GPU {i}: {gpu_props.name} ({memory_gb:.1f}GB)")
        
        # Log distributed training info
        if dist.is_available() and dist.is_initialized():
            self.multigpu_logger.info(f"Distributed backend: {dist.get_backend()}")
            self.multigpu_logger.info(f"World size: {dist.get_world_size()}")
            self.multigpu_logger.info(f"Local rank: {dist.get_rank()}")
    
    def log_training_step(self, step: int, loss: float, lr: float, metrics: Dict = None):
        """Log training step with readable format"""
        
        # Calculate progress
        progress = (step / self.session.max_steps) * 100
        
        # Log every N steps or at milestones
        should_log = (
            step % 50 == 0 or  # Every 50 steps
            step < 10 or       # First 10 steps
            step % (self.session.max_steps // 10) == 0  # 10% milestones
        )
        
        if should_log and self.accelerator.is_main_process:
            elapsed = (datetime.now() - self.session.start_time).total_seconds()
            steps_per_sec = step / elapsed if elapsed > 0 else 0
            eta_seconds = (self.session.max_steps - step) / steps_per_sec if steps_per_sec > 0 else 0
            eta_str = f"{eta_seconds/60:.1f}min" if eta_seconds < 3600 else f"{eta_seconds/3600:.1f}h"
            
            self.main_logger.info(
                f"Step {step:4d}/{self.session.max_steps} ({progress:5.1f}%) | "
                f"Loss: {loss:.4f} | LR: {lr:.2e} | "
                f"Speed: {steps_per_sec:.2f} steps/s | ETA: {eta_str}"
            )
            
            # Log additional metrics if provided
            if metrics:
                metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                self.main_logger.info(f"  └─ Metrics: {metric_str}")
        
        # Store metrics for analysis
        self.training_metrics.append({
            "step": step,
            "loss": loss,
            "learning_rate": lr,
            "timestamp": datetime.now().isoformat(),
            "process_rank": self.accelerator.local_process_index,
            "metrics": metrics or {}
        })
    
    def log_evaluation(self, step: int, eval_metrics: Dict):
        """Log evaluation results"""
        
        if self.accelerator.is_main_process:
            self.main_logger.info("\n📊 EVALUATION RESULTS:")
            for metric, value in eval_metrics.items():
                self.main_logger.info(f"  • {metric}: {value:.4f}")
            self.main_logger.info("")
    
    def log_gpu_communication(self, operation: str, data_size_mb: float = None, duration_ms: float = None):
        """Log GPU communication operations"""
        
        comm_info = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "process_rank": self.accelerator.local_process_index,
            "data_size_mb": data_size_mb,
            "duration_ms": duration_ms
        }
        
        self.communication_logs.append(comm_info)
        
        if data_size_mb and duration_ms:
            bandwidth = data_size_mb / (duration_ms / 1000)  # MB/s
            self.multigpu_logger.info(
                f"Communication | {operation} | Rank {self.accelerator.local_process_index} | "
                f"Size: {data_size_mb:.2f}MB | Duration: {duration_ms:.2f}ms | "
                f"Bandwidth: {bandwidth:.2f}MB/s"
            )
    
    def start_background_monitoring(self):
        """Start background GPU monitoring"""
        
        def monitor_loop():
            while self.monitoring:
                try:
                    snapshot = self.capture_gpu_snapshot()
                    self.gpu_snapshots.append(snapshot)
                    
                    # Log performance issues
                    self.check_performance_issues(snapshot)
                    
                    time.sleep(10)  # Monitor every 10 seconds
                except Exception as e:
                    self.main_logger.error(f"Monitoring error: {e}")
                    time.sleep(5)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def capture_gpu_snapshot(self) -> Dict:
        """Capture current GPU state"""
        
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "gpus": [],
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
            }
        }
        
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                snapshot["gpus"].append({
                    "id": gpu.id,
                    "name": gpu.name,
                    "utilization": gpu.load * 100,
                    "memory_used_mb": gpu.memoryUsed,
                    "memory_total_mb": gpu.memoryTotal,
                    "memory_percent": gpu.memoryUtil * 100,
                    "temperature": gpu.temperature,
                })
        except Exception as e:
            self.main_logger.error(f"GPU snapshot error: {e}")
        
        return snapshot
    
    def check_performance_issues(self, snapshot: Dict):
        """Check for performance issues and log warnings"""
        
        for gpu in snapshot["gpus"]:
            # Low GPU utilization warning
            if gpu["utilization"] < 50:
                self.perf_logger.warning(
                    f"GPU {gpu['id']} utilization low: {gpu['utilization']:.1f}%"
                )
            
            # High memory usage warning
            if gpu["memory_percent"] > 90:
                self.perf_logger.warning(
                    f"GPU {gpu['id']} memory high: {gpu['memory_percent']:.1f}%"
                )
            
            # Temperature warnings
            if gpu["temperature"] > 80:
                self.perf_logger.warning(
                    f"GPU {gpu['id']} temperature high: {gpu['temperature']}°C"
                )
    
    def log_epoch_summary(self, epoch: int, avg_loss: float, eval_loss: float = None):
        """Log epoch summary"""
        
        if self.accelerator.is_main_process:
            self.main_logger.info("\n" + "="*60)
            self.main_logger.info(f"📈 EPOCH {epoch} SUMMARY")
            self.main_logger.info(f"  • Average Loss: {avg_loss:.4f}")
            if eval_loss:
                self.main_logger.info(f"  • Evaluation Loss: {eval_loss:.4f}")
            self.main_logger.info("="*60 + "\n")
    
    def finalize_session(self):
        """Finalize logging session and save summary"""
        
        self.monitoring = False
        
        if self.accelerator.is_main_process:
            self.main_logger.info("\n" + "="*80)
            self.main_logger.info("🎯 TRAINING SESSION COMPLETED")
            self.main_logger.info("="*80)
            
            # Calculate session statistics
            duration = datetime.now() - self.session.start_time
            total_steps = len(self.training_metrics)
            
            self.main_logger.info(f"Session Duration: {duration}")
            self.main_logger.info(f"Total Steps: {total_steps}")
            self.main_logger.info(f"Average Speed: {total_steps / duration.total_seconds():.2f} steps/sec")
            
            # Save comprehensive summary
            self.save_session_summary()
            
            self.main_logger.info(f"Session logs saved to: {self.log_dir}")
            self.main_logger.info("="*80)
    
    def save_session_summary(self):
        """Save comprehensive session summary"""
        
        summary = {
            "session": {
                "id": self.session_id,
                "start_time": self.session.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": (datetime.now() - self.session.start_time).total_seconds(),
                "mode": self.session.mode,
                "gpu_count": self.session.gpu_count,
                "model_name": self.session.model_name,
                "dataset_name": self.session.dataset_name,
            },
            "training_metrics": self.training_metrics,
            "gpu_snapshots": self.gpu_snapshots[-100:],  # Last 100 snapshots
            "communication_logs": self.communication_logs,
        }
        
        # Save detailed summary
        summary_path = os.path.join(self.log_dir, f"{self.session_id}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save readable summary
        readable_path = os.path.join(self.log_dir, f"{self.session_id}_summary.txt")
        self.create_readable_summary(readable_path, summary)
    
    def create_readable_summary(self, filepath: str, summary: Dict):
        """Create human-readable summary file"""
        
        with open(filepath, 'w') as f:
            f.write("="*80 + "\n")
            f.write("TRAINING SESSION SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            # Session info
            f.write("SESSION INFORMATION:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Session ID: {summary['session']['id']}\n")
            f.write(f"Mode: {summary['session']['mode'].upper()}\n")
            f.write(f"GPU Count: {summary['session']['gpu_count']}\n")
            f.write(f"Duration: {summary['session']['duration_seconds']:.1f} seconds\n")
            f.write(f"Model: {summary['session']['model_name']}\n")
            f.write(f"Dataset: {summary['session']['dataset_name']}\n\n")
            
            # Training metrics summary
            if summary['training_metrics']:
                metrics = summary['training_metrics']
                f.write("TRAINING METRICS:\n")
                f.write("-" * 17 + "\n")
                f.write(f"Total Steps: {len(metrics)}\n")
                f.write(f"Final Loss: {metrics[-1]['loss']:.4f}\n")
                f.write(f"Initial Loss: {metrics[0]['loss']:.4f}\n")
                f.write(f"Loss Improvement: {metrics[0]['loss'] - metrics[-1]['loss']:.4f}\n\n")
            
            # GPU utilization summary
            if summary['gpu_snapshots']:
                f.write("GPU UTILIZATION SUMMARY:\n")
                f.write("-" * 25 + "\n")
                
                # Calculate averages per GPU
                gpu_count = len(summary['gpu_snapshots'][0]['gpus'])
                for gpu_id in range(gpu_count):
                    utilizations = [s['gpus'][gpu_id]['utilization'] for s in summary['gpu_snapshots']]
                    memory_usages = [s['gpus'][gpu_id]['memory_percent'] for s in summary['gpu_snapshots']]
                    
                    avg_util = sum(utilizations) / len(utilizations)
                    max_util = max(utilizations)
                    avg_memory = sum(memory_usages) / len(memory_usages)
                    max_memory = max(memory_usages)
                    
                    f.write(f"GPU {gpu_id}:\n")
                    f.write(f"  Average Utilization: {avg_util:.1f}%\n")
                    f.write(f"  Peak Utilization: {max_util:.1f}%\n")
                    f.write(f"  Average Memory: {avg_memory:.1f}%\n")
                    f.write(f"  Peak Memory: {max_memory:.1f}%\n\n")
            
            f.write("="*80 + "\n")


# Integration helper functions
def create_training_logger(config, accelerator: Accelerator) -> MultiGPUTrainingLogger:
    """Create and return a training logger instance"""
    return MultiGPUTrainingLogger(config, accelerator)


def log_model_parameters(logger: MultiGPUTrainingLogger, model):
    """Log model parameter information"""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.main_logger.info(f"\n🧠 MODEL PARAMETERS:")
    logger.main_logger.info(f"  • Total parameters: {total_params:,}")
    logger.main_logger.info(f"  • Trainable parameters: {trainable_params:,}")
    logger.main_logger.info(f"  • Trainable percentage: {100 * trainable_params / total_params:.2f}%")


def log_dataset_info(logger: MultiGPUTrainingLogger, train_dataset, eval_dataset):
    """Log dataset information"""
    
    logger.main_logger.info(f"\n📚 DATASET INFORMATION:")
    logger.main_logger.info(f"  • Training samples: {len(train_dataset):,}")
    logger.main_logger.info(f"  • Evaluation samples: {len(eval_dataset):,}")
    logger.main_logger.info(f"  • Total samples: {len(train_dataset) + len(eval_dataset):,}")