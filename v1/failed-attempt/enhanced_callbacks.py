#!/usr/bin/env python3

import time
import torch
import torch.distributed as dist
from typing import Dict, Optional
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from training_logger import MultiGPUTrainingLogger


class EnhancedTrainingCallback(TrainerCallback):
    """Enhanced callback that integrates with our custom logging system"""
    
    def __init__(self, training_logger: MultiGPUTrainingLogger):
        self.training_logger = training_logger
        self.step_start_time = None
        self.epoch_start_time = None
        self.epoch_losses = []
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, 
                      control: TrainerControl, **kwargs):
        """Called at the beginning of training"""
        self.training_logger.main_logger.info("🚀 Training started...")
        
        # Log distributed training info if applicable
        if dist.is_available() and dist.is_initialized():
            self.training_logger.log_gpu_communication(
                "training_start", 
                data_size_mb=0,
                duration_ms=0
            )
    
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState,
                      control: TrainerControl, **kwargs):
        """Called at the beginning of each epoch"""
        self.epoch_start_time = time.time()
        self.epoch_losses = []
        
        if self.training_logger.accelerator.is_main_process:
            self.training_logger.main_logger.info(
                f"\n📚 Starting Epoch {state.epoch + 1}"
            )
    
    def on_step_begin(self, args: TrainingArguments, state: TrainerState,
                     control: TrainerControl, **kwargs):
        """Called at the beginning of each training step"""
        self.step_start_time = time.time()
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState,
                   control: TrainerControl, logs: Optional[Dict[str, float]] = None, **kwargs):
        """Called at the end of each training step"""
        
        if logs and self.step_start_time:
            step_duration = time.time() - self.step_start_time
            
            # Extract metrics
            loss = logs.get("train_loss", 0.0)
            lr = logs.get("learning_rate", 0.0)
            
            # Log training step
            self.training_logger.log_training_step(
                step=state.global_step,
                loss=loss,
                lr=lr,
                metrics={
                    "step_duration": step_duration,
                    "samples_per_second": args.per_device_train_batch_size / step_duration,
                }
            )
            
            # Store loss for epoch summary
            self.epoch_losses.append(loss)
            
            # Log GPU communication for gradient synchronization
            if dist.is_available() and dist.is_initialized() and state.global_step % 10 == 0:
                # Estimate gradient sync data size (rough approximation)
                total_params = sum(p.numel() for p in kwargs.get("model", {}).parameters() if p.requires_grad)
                grad_sync_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
                
                self.training_logger.log_gpu_communication(
                    "gradient_sync",
                    data_size_mb=grad_sync_mb,
                    duration_ms=step_duration * 1000
                )
    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState,
                   control: TrainerControl, logs: Optional[Dict[str, float]] = None, **kwargs):
        """Called after evaluation"""
        
        if logs:
            # Filter evaluation metrics
            eval_metrics = {k: v for k, v in logs.items() if k.startswith("eval_")}
            
            self.training_logger.log_evaluation(
                step=state.global_step,
                eval_metrics=eval_metrics
            )
    
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, logs: Optional[Dict[str, float]] = None, **kwargs):
        """Called at the end of each epoch"""
        
        if self.epoch_losses and self.epoch_start_time:
            avg_loss = sum(self.epoch_losses) / len(self.epoch_losses)
            eval_loss = logs.get("eval_loss") if logs else None
            
            self.training_logger.log_epoch_summary(
                epoch=int(state.epoch),
                avg_loss=avg_loss,
                eval_loss=eval_loss
            )
            
            # Log epoch timing
            epoch_duration = time.time() - self.epoch_start_time
            if self.training_logger.accelerator.is_main_process:
                self.training_logger.main_logger.info(
                    f"📊 Epoch {int(state.epoch)} completed in {epoch_duration:.1f}s"
                )
    
    def on_save(self, args: TrainingArguments, state: TrainerState,
               control: TrainerControl, **kwargs):
        """Called when saving a checkpoint"""
        
        if self.training_logger.accelerator.is_main_process:
            self.training_logger.main_logger.info(
                f"💾 Checkpoint saved at step {state.global_step}"
            )
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, **kwargs):
        """Called at the end of training"""
        
        self.training_logger.main_logger.info("🎯 Training completed!")
        
        # Finalize logging session
        self.training_logger.finalize_session()


class MultiGPUMonitoringCallback(TrainerCallback):
    """Callback specifically for monitoring multi-GPU behavior"""
    
    def __init__(self, training_logger: MultiGPUTrainingLogger):
        self.training_logger = training_logger
        self.gradient_sync_times = []
        self.communication_times = []
    
    def on_step_begin(self, args: TrainingArguments, state: TrainerState,
                     control: TrainerControl, **kwargs):
        """Monitor step begin for communication patterns"""
        
        if dist.is_available() and dist.is_initialized():
            # Monitor if we're using multiple GPUs
            world_size = dist.get_world_size()
            if world_size > 1:
                self.training_logger.multigpu_logger.debug(
                    f"Step {state.global_step} - Multi-GPU sync begin | "
                    f"World size: {world_size} | Rank: {dist.get_rank()}"
                )
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState,
                   control: TrainerControl, **kwargs):
        """Monitor step end for gradient synchronization"""
        
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1 and state.global_step % 20 == 0:  # Log every 20 steps
                
                # Log multi-GPU coordination
                self.training_logger.multigpu_logger.info(
                    f"Multi-GPU Step {state.global_step} | "
                    f"Processes: {world_size} | "
                    f"Local rank: {self.training_logger.accelerator.local_process_index} | "
                    f"Process index: {self.training_logger.accelerator.process_index}"
                )
                
                # Monitor memory across GPUs
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                        memory_reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
                        
                        self.training_logger.multigpu_logger.debug(
                            f"GPU {i} Memory | Allocated: {memory_allocated:.2f}GB | "
                            f"Reserved: {memory_reserved:.2f}GB"
                        )


class PerformanceMonitoringCallback(TrainerCallback):
    """Callback for monitoring training performance and bottlenecks"""
    
    def __init__(self, training_logger: MultiGPUTrainingLogger):
        self.training_logger = training_logger
        self.step_times = []
        self.last_log_time = time.time()
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState,
                   control: TrainerControl, **kwargs):
        """Monitor performance metrics"""
        
        current_time = time.time()
        
        # Log performance summary every 100 steps
        if state.global_step % 100 == 0 and state.global_step > 0:
            
            time_since_last = current_time - self.last_log_time
            steps_in_period = 100 if state.global_step >= 100 else state.global_step
            avg_step_time = time_since_last / steps_in_period
            
            # Calculate throughput
            total_batch_size = (
                args.per_device_train_batch_size * 
                args.gradient_accumulation_steps * 
                self.training_logger.accelerator.num_processes
            )
            samples_per_second = total_batch_size / avg_step_time
            
            # Log performance metrics
            self.training_logger.perf_logger.info(
                f"Performance | Step {state.global_step} | "
                f"Avg step time: {avg_step_time:.3f}s | "
                f"Throughput: {samples_per_second:.1f} samples/s | "
                f"Total batch size: {total_batch_size}"
            )
            
            # Reset timer
            self.last_log_time = current_time
            
            # Check for performance issues
            if avg_step_time > 10.0:  # If steps are taking too long
                self.training_logger.perf_logger.warning(
                    f"Slow training detected: {avg_step_time:.1f}s per step"
                )
            
            if samples_per_second < 1.0:  # Very low throughput
                self.training_logger.perf_logger.warning(
                    f"Low throughput detected: {samples_per_second:.2f} samples/s"
                )