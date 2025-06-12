from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class Config:
    # GPU Configuration
    use_multi_gpu: bool = True  # Toggle for multi-GPU vs single GPU
    force_single_gpu: bool = False  # Force single GPU even if multiple available
    gpu_device_ids: Optional[list] = None  # Specific GPU IDs to use (None = all available)
    
    # Model Configuration
    model_name: str = "unsloth/llama-2-7b-bnb-4bit"  # Working Unsloth model
    max_seq_length: int = 512  # Reduced for stability
    dtype: Optional[torch.dtype] = None  # Auto-detected
    load_in_4bit: bool = True
    
    # Dataset Configuration
    dataset_name: str = "timdettmers/openassistant-guanaco"
    dataset_text_field: str = "text"
    dataset_split: str = "train"
    max_samples: Optional[int] = None  # Use all samples if None
    
    # Training Configuration
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    max_steps: int = 50  # Short for testing -- 1000
    learning_rate: float = 2e-4
    fp16: bool = False
    bf16: bool = True
    logging_steps: int = 1
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 42
    
    # LoRA Configuration
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: list = None
    
    # Evaluation Configuration
    eval_steps: int = 25  # Match with short training (50 steps)
    eval_strategy: str = "steps"
    per_device_eval_batch_size: int = 2
    save_best_model: bool = True
    
    # Output Configuration
    output_dir: str = "./checkpoints"
    logging_dir: str = "./logs"
    save_steps: int = 25  # Match eval_steps for save_best_model
    save_total_limit: int = 3
    resume_from_checkpoint: Optional[str] = None
    
    # Accelerate Configuration
    mixed_precision: str = "bf16"  # or "fp16"
    gradient_checkpointing: bool = True
    dataloader_pin_memory: bool = False  # Disable for stability
    dataloader_num_workers: int = 0  # Single threaded for stability
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                                 "gate_proj", "up_proj", "down_proj"]
        
        if self.dtype is None:
            self.dtype = torch.bfloat16 if self.bf16 else torch.float16
        
        # Ensure output directories exist
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logging_dir, exist_ok=True)