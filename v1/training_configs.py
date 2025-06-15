#!/usr/bin/env python3

"""
Training Configuration System for Multi-GPU QLoRA Fine-Tuning
Provides small, medium, and large configuration presets for different training scales
"""

import argparse
import sys

TRAINING_CONFIGS = {
    "small": {
        # Model Configuration
        "model_name": "microsoft/DialoGPT-small",  # 117M parameters
        "model_type": "standard",  # standard, unsloth
        "max_seq_length": 256,
        "quantization": "8bit",  # none, 8bit, 4bit
        "use_bnb": True,  # Use BitsAndBytes quantization
        
        # Training Configuration
        "max_steps": 20,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 2,
        "learning_rate": 5e-4,
        "warmup_steps": 2,
        
        # LoRA Configuration
        "use_lora": True,
        "lora_method": "peft",  # peft, unsloth
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "target_modules": ["c_attn", "c_proj"],  # DialoGPT specific
        
        # Dataset Configuration
        "dataset_name": "tatsu-lab/alpaca",
        "dataset_config": None,
        "dataset_split": "train[:100]",  # Very small for testing
        "dataset_text_field": "text",
        
        # Optimization
        "optimizer": "adamw_torch",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        
        # Output Configuration
        "eval_strategy": "no",
        "save_steps": 20,
        "logging_steps": 1,
    },
    
    "medium": {
        # Model Configuration
        "model_name": "microsoft/DialoGPT-medium",  # 345M parameters
        "model_type": "standard",
        "max_seq_length": 512,
        "quantization": "8bit",
        "use_bnb": True,  # Use BitsAndBytes quantization
        
        # Training Configuration
        "max_steps": 100,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "warmup_steps": 10,
        
        # LoRA Configuration
        "use_lora": True,
        "lora_method": "peft",
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": ["c_attn", "c_proj"],
        
        # Dataset Configuration
        "dataset_name": "tatsu-lab/alpaca",
        "dataset_config": None,
        "dataset_split": "train[:1000]",
        "dataset_text_field": "text",
        
        # Optimization
        "optimizer": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        
        # Output Configuration
        "eval_strategy": "steps",
        "eval_steps": 50,
        "save_steps": 50,
        "logging_steps": 5,
    },
    
    "large": {
        # Model Configuration
        "model_name": "unsloth/llama-2-7b-bnb-4bit",  # 7B parameters
        "model_type": "unsloth",  # Use Unsloth for large models
        "max_seq_length": 1024,
        "quantization": "4bit",
        "use_bnb": True,  # Use BitsAndBytes quantization
        
        # Training Configuration
        "max_steps": 500,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,  # Smaller due to memory
        "gradient_accumulation_steps": 8,
        "learning_rate": 1e-4,
        "warmup_steps": 50,
        
        # LoRA Configuration
        "use_lora": True,
        "lora_method": "unsloth",
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        
        # Dataset Configuration
        "dataset_name": "timdettmers/openassistant-guanaco",
        "dataset_config": None,
        "dataset_split": "train[:5000]",
        "dataset_text_field": "text",
        
        # Optimization
        "optimizer": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        
        # Output Configuration
        "eval_strategy": "steps",
        "eval_steps": 100,
        "save_steps": 100,
        "logging_steps": 10,
    }
}

def get_config(config_name="small"):
    """
    Get training configuration by name
    
    Args:
        config_name (str): Configuration name ('small', 'medium', 'large')
        
    Returns:
        dict: Training configuration parameters
    """
    if config_name not in TRAINING_CONFIGS:
        print(f"Error: Unknown config '{config_name}'. Available configs: {list(TRAINING_CONFIGS.keys())}")
        sys.exit(1)
        
    config = TRAINING_CONFIGS[config_name].copy()
    
    # Print configuration details
    print(f"\n=== {config_name.upper()} TRAINING CONFIGURATION ===")
    print(f"Model: {config['model_name']}")
    print(f"Dataset: {config['dataset_name']}")
    print(f"Training Steps: {config['max_steps']}")
    print(f"Quantization: {config['quantization']} with BitsAndBytes")
    print(f"LoRA Config: r={config['r']}, alpha={config['lora_alpha']}, dropout={config['lora_dropout']}")
    print(f"Training Mechanism: QLoRA (Quantized Low-Rank Adaptation)")
    print("=" * 50)
    
    return config

def parse_training_args():
    """Parse command line arguments for training configuration"""
    parser = argparse.ArgumentParser(description="Multi-GPU QLoRA Fine-Tuning")
    parser.add_argument(
        "--config", 
        type=str, 
        default="small",
        choices=["small", "medium", "large"],
        help="Training configuration preset (default: small)"
    )
    parser.add_argument(
        "--trainsession",
        type=str,
        default="default",
        help="Training session name for organized output folders (default: default)"
    )
    parser.add_argument(
        "--force-single-gpu",
        action="store_true",
        help="Force single GPU training even in multi-GPU environment"
    )
    return parser.parse_args()

def print_config_summary(config, size):
    """Print a nice summary of the configuration"""
    print("="*70)
    print(f"🚀 TRAINING CONFIGURATION: {size.upper()}")
    print("="*70)
    
    print(f"\n📱 MODEL CONFIGURATION:")
    print(f"  Model: {config['model_name']}")
    print(f"  Type: {config['model_type']}")
    print(f"  Max Sequence Length: {config['max_seq_length']}")
    print(f"  Quantization: {config['quantization']}")
    
    print(f"\n🎯 TRAINING CONFIGURATION:")
    print(f"  Method: QLoRA (LoRA + Quantization)")
    print(f"  Max Steps: {config['max_steps']}")
    print(f"  Batch Size (per device): {config['per_device_train_batch_size']}")
    print(f"  Gradient Accumulation: {config['gradient_accumulation_steps']}")
    print(f"  Learning Rate: {config['learning_rate']}")
    print(f"  Optimizer: {config['optimizer']}")
    
    print(f"\n🔧 LORA CONFIGURATION:")
    print(f"  Method: {config['lora_method']}")
    print(f"  Rank (r): {config['r']}")
    print(f"  Alpha: {config['lora_alpha']}")
    print(f"  Dropout: {config['lora_dropout']}")
    print(f"  Target Modules: {config['target_modules']}")
    
    print(f"\n📚 DATASET CONFIGURATION:")
    print(f"  Dataset: {config['dataset_name']}")
    if config['dataset_config']:
        print(f"  Config: {config['dataset_config']}")
    print(f"  Split: {config['dataset_split']}")
    
    # Calculate total batch size for multi-GPU
    total_batch_size_single = config['per_device_train_batch_size'] * config['gradient_accumulation_steps']
    total_batch_size_multi = total_batch_size_single * 2  # Assuming 2 GPUs
    
    print(f"\n⚡ EXPECTED PERFORMANCE:")
    print(f"  Single GPU Total Batch Size: {total_batch_size_single}")
    print(f"  Multi-GPU Total Batch Size: {total_batch_size_multi}")
    
    # Estimate parameters based on model
    if "small" in config['model_name'].lower() or "117" in config['model_name']:
        params = "~117M"
    elif "medium" in config['model_name'].lower() or "345" in config['model_name']:
        params = "~345M"
    elif "7b" in config['model_name'].lower():
        params = "~7B"
    else:
        params = "Unknown"
    
    print(f"  Model Parameters: {params}")
    
    if config['quantization'] == "4bit":
        quant_desc = "4-bit (most memory efficient)"
    elif config['quantization'] == "8bit":
        quant_desc = "8-bit (balanced)"
    else:
        quant_desc = "Full precision (highest quality)"
    
    print(f"  Quantization: {quant_desc}")
    
    print("="*70)

def get_model_info(config):
    """Get detailed model information"""
    model_info = {
        "microsoft/DialoGPT-small": {
            "parameters": "117M",
            "type": "Conversational AI",
            "memory_fp16": "~0.5GB",
            "memory_8bit": "~0.3GB"
        },
        "microsoft/DialoGPT-medium": {
            "parameters": "345M", 
            "type": "Conversational AI",
            "memory_fp16": "~1.4GB",
            "memory_8bit": "~0.7GB"
        },
        "unsloth/llama-2-7b-bnb-4bit": {
            "parameters": "7B",
            "type": "Large Language Model",
            "memory_fp16": "~14GB",
            "memory_4bit": "~3.5GB"
        }
    }
    
    return model_info.get(config['model_name'], {
        "parameters": "Unknown",
        "type": "Language Model",
        "memory_fp16": "Unknown",
        "memory_4bit": "Unknown"
    })

if __name__ == "__main__":
    # Demo: Show all available configurations
    print("Available Training Configurations:")
    print("=" * 50)
    
    for config_name in TRAINING_CONFIGS.keys():
        config = get_config(config_name)
        print()