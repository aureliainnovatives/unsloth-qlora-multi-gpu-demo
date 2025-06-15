import os
import json
import logging
import torch
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from accelerate.logging import get_logger
import pickle

logger = get_logger(__name__)


def setup_logging(accelerator=None):
    """Setup logging configuration"""
    
    # Create logs directory
    os.makedirs("./logs", exist_ok=True)
    
    # Configure logging
    log_level = logging.INFO
    if accelerator and not accelerator.is_main_process:
        log_level = logging.WARNING
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=log_level,
        handlers=[
            logging.FileHandler(f"./logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    
    # Set specific loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("accelerate").setLevel(logging.INFO)
    logging.getLogger("datasets").setLevel(logging.WARNING)


def save_config(config, output_dir: str):
    """Save config to JSON file"""
    config_dict = {}
    for key, value in config.__dict__.items():
        if isinstance(value, torch.dtype):
            config_dict[key] = str(value)
        elif hasattr(value, '__name__'):  # For functions/classes
            config_dict[key] = value.__name__
        else:
            config_dict[key] = value
    
    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    
    logger.info(f"Config saved to {config_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load config from JSON file"""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return config_dict


def print_trainable_parameters(model):
    """Print the number of trainable parameters in the model"""
    trainable_params = 0
    all_param = 0
    
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    percentage = 100 * trainable_params / all_param
    
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"All parameters: {all_param:,}")
    logger.info(f"Trainable percentage: {percentage:.4f}%")
    
    return trainable_params, all_param, percentage


def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    
    # Calculate perplexity
    # Note: predictions are logits, labels are token ids
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # Flatten predictions and labels
    predictions = predictions.reshape(-1, predictions.shape[-1])
    labels = labels.reshape(-1)
    
    # Remove padding tokens (-100)
    valid_mask = labels != -100
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    
    # Calculate loss manually
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(torch.tensor(valid_predictions), torch.tensor(valid_labels))
    perplexity = torch.exp(loss).item()
    
    return {
        "eval_perplexity": perplexity,
        "eval_loss": loss.item(),
    }


class EvaluationCallback(TrainerCallback):
    """Custom callback for enhanced evaluation logging"""
    
    def __init__(self):
        self.best_eval_loss = float('inf')
        self.eval_history = []
    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, 
                   control: TrainerControl, model=None, logs: Optional[Dict[str, float]] = None, **kwargs):
        """Called after evaluation"""
        if logs:
            eval_loss = logs.get("eval_loss", float('inf'))
            eval_perplexity = logs.get("eval_perplexity", float('inf'))
            
            # Track evaluation history
            self.eval_history.append({
                "step": state.global_step,
                "eval_loss": eval_loss,
                "eval_perplexity": eval_perplexity,
                "timestamp": datetime.now().isoformat()
            })
            
            # Check if this is the best model
            if eval_loss < self.best_eval_loss:
                self.best_eval_loss = eval_loss
                logger.info(f"New best model! Eval loss: {eval_loss:.4f}, Perplexity: {eval_perplexity:.4f}")
            
            # Log additional metrics
            logger.info(f"Step {state.global_step} - Eval Loss: {eval_loss:.4f}, Perplexity: {eval_perplexity:.4f}")
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, 
                    control: TrainerControl, **kwargs):
        """Called at the end of training"""
        # Save evaluation history
        history_path = os.path.join(args.output_dir, "eval_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.eval_history, f, indent=2)
        
        logger.info(f"Evaluation history saved to {history_path}")
        logger.info(f"Best evaluation loss: {self.best_eval_loss:.4f}")


class MemoryTracker:
    """Track GPU memory usage during training"""
    
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory_history = []
        self.peak_memory = 0
    
    def track_memory(self, step: int, phase: str = ""):
        """Track current memory usage"""
        if self.device.type == "cuda":
            current_memory = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
            max_memory = torch.cuda.max_memory_allocated(self.device) / 1024**3  # GB
            
            self.memory_history.append({
                "step": step,
                "phase": phase,
                "current_memory_gb": current_memory,
                "max_memory_gb": max_memory,
                "timestamp": datetime.now().isoformat()
            })
            
            if max_memory > self.peak_memory:
                self.peak_memory = max_memory
            
            logger.debug(f"Step {step} {phase} - Memory: {current_memory:.2f}GB (Peak: {max_memory:.2f}GB)")
    
    def save_memory_log(self, output_dir: str):
        """Save memory tracking log"""
        memory_log_path = os.path.join(output_dir, "memory_log.json")
        with open(memory_log_path, 'w') as f:
            json.dump({
                "peak_memory_gb": self.peak_memory,
                "history": self.memory_history
            }, f, indent=2)
        
        logger.info(f"Memory log saved to {memory_log_path}")
        logger.info(f"Peak memory usage: {self.peak_memory:.2f}GB")


def save_checkpoint_metadata(checkpoint_dir: str, step: int, metrics: Dict[str, float]):
    """Save additional metadata for checkpoints"""
    metadata = {
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "torch_version": torch.__version__,
    }
    
    metadata_path = os.path.join(checkpoint_dir, "checkpoint_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def calculate_model_size(model):
    """Calculate model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def get_gpu_info():
    """Get GPU information"""
    if not torch.cuda.is_available():
        return {"gpu_available": False}
    
    gpu_info = {
        "gpu_available": True,
        "gpu_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "devices": []
    }
    
    for i in range(torch.cuda.device_count()):
        device_props = torch.cuda.get_device_properties(i)
        gpu_info["devices"].append({
            "id": i,
            "name": device_props.name,
            "memory_total_gb": device_props.total_memory / 1024**3,
            "memory_free_gb": (device_props.total_memory - torch.cuda.memory_allocated(i)) / 1024**3,
            "compute_capability": f"{device_props.major}.{device_props.minor}"
        })
    
    return gpu_info


def create_training_summary(config, train_result, eval_result, output_dir: str):
    """Create a comprehensive training summary"""
    summary = {
        "training_config": {
            "model_name": config.model_name,
            "max_seq_length": config.max_seq_length,
            "batch_size": config.per_device_train_batch_size,
            "learning_rate": config.learning_rate,
            "max_steps": config.max_steps,
            "lora_r": config.r,
            "lora_alpha": config.lora_alpha,
        },
        "training_results": {
            "final_train_loss": train_result.training_loss if hasattr(train_result, 'training_loss') else None,
            "final_eval_loss": eval_result.get("eval_loss") if eval_result else None,
            "final_perplexity": eval_result.get("eval_perplexity") if eval_result else None,
            "total_steps": train_result.global_step if hasattr(train_result, 'global_step') else None,
        },
        "system_info": get_gpu_info(),
        "timestamp": datetime.now().isoformat(),
    }
    
    summary_path = os.path.join(output_dir, "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Training summary saved to {summary_path}")
    return summary