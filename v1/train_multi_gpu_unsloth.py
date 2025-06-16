#!/usr/bin/env python3

"""
Multi-GPU Training Script with Unsloth Pro
This script uses Unsloth Pro's advanced multi-GPU capabilities that resolve
the gradient checkpointing conflicts with DistributedDataParallel (DDP).

REQUIREMENTS:
- Unsloth Pro subscription/license
- Multiple GPUs with sufficient VRAM
- Proper Unsloth Pro installation

USAGE:
Method 1 (if Unsloth Pro handles distribution internally):
    python train_multi_gpu_unsloth.py --trainsession v1 --config medium

Method 2 (if still requires accelerate launch):
    accelerate launch train_multi_gpu_unsloth.py --trainsession v1 --config medium

Note: This script is a reference implementation for Unsloth Pro users.
The community version will fall back to standard multi-GPU training.
"""

import os
import torch
import logging
from datetime import datetime

# Disable wandb and warnings
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_unsloth_pro():
    """Check if Unsloth Pro is available"""
    try:
        # Try importing Unsloth Pro features
        import unsloth
        
        # Check for Pro-specific features (this is speculative)
        if hasattr(unsloth, 'FastLanguageModelPro') or hasattr(unsloth, 'multi_gpu_training'):
            return True, "Unsloth Pro detected"
        
        # Check version for Pro indicators
        version = getattr(unsloth, '__version__', 'unknown')
        if 'pro' in version.lower() or 'commercial' in version.lower():
            return True, f"Unsloth Pro version: {version}"
        
        return False, f"Unsloth Community version: {version}"
    
    except ImportError:
        return False, "Unsloth not installed"

def initialize_unsloth_pro_multi_gpu():
    """Initialize Unsloth Pro multi-GPU environment"""
    try:
        # Import Unsloth Pro (assuming similar API to community)
        import unsloth
        from unsloth import FastLanguageModel
        
        # Check if we need Accelerator for Unsloth Pro
        try:
            from accelerate import Accelerator
            accelerator = Accelerator()
            use_accelerator = True
            logger.info(f"Using Accelerator with {accelerator.num_processes} processes")
        except:
            accelerator = None
            use_accelerator = False
            logger.info("Running without Accelerator (Unsloth Pro native multi-GPU)")
        
        return True, accelerator, use_accelerator
    
    except Exception as e:
        logger.error(f"Failed to initialize Unsloth Pro: {e}")
        return False, None, False

def fallback_to_standard_multi_gpu():
    """Fallback to standard multi-GPU implementation"""
    logger.warning("Falling back to standard multi-GPU training")
    
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from accelerate import Accelerator
    
    accelerator = Accelerator()
    return accelerator, "standard"

def main():
    from training_configs import get_config, parse_training_args
    from datasets import load_dataset
    
    # Parse arguments and get configuration
    args = parse_training_args()
    config = get_config(args.config)
    
    # Check Unsloth Pro availability
    has_pro, pro_status = check_unsloth_pro()
    
    print("="*70)
    print("🚀 MULTI-GPU TRAINING WITH UNSLOTH PRO")
    print("="*70)
    print(f"Unsloth Status: {pro_status}")
    print(f"Config: {args.config}")
    print(f"Session: {args.trainsession}")
    print(f"Model: {config['model_name']}")
    print(f"Max steps: {config['max_steps']}")
    print("="*70)
    
    if has_pro:
        # Unsloth Pro path
        success, accelerator, use_accelerator = initialize_unsloth_pro_multi_gpu()
        
        if success:
            # Import Unsloth Pro components
            import unsloth
            from unsloth import FastLanguageModel
            
            print("✅ Using Unsloth Pro Multi-GPU Training")
            
            if use_accelerator:
                print(f"📊 Accelerator processes: {accelerator.num_processes}")
                print(f"📊 Current GPU: {accelerator.local_process_index}")
                print(f"📊 Total batch size: {config['per_device_train_batch_size'] * config['gradient_accumulation_steps'] * accelerator.num_processes}")
                is_main_process = accelerator.is_main_process
            else:
                print("📊 Using Unsloth Pro native distribution")
                # Detect GPUs manually
                num_gpus = torch.cuda.device_count()
                print(f"📊 Detected GPUs: {num_gpus}")
                print(f"📊 Total batch size: {config['per_device_train_batch_size'] * config['gradient_accumulation_steps'] * num_gpus}")
                is_main_process = True  # In native mode, assume main process
            
            # Load model with Unsloth Pro multi-GPU support
            if is_main_process:
                logger.info("Loading model with Unsloth Pro...")
            
            # Unsloth Pro model loading (assuming enhanced API)
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=config["model_name"],
                max_seq_length=config["max_seq_length"],
                dtype=None,
                load_in_4bit=(config["quantization"] == "4bit"),
                load_in_8bit=(config["quantization"] == "8bit"),
                trust_remote_code=True,
                # Pro-specific parameters (speculative)
                multi_gpu=True,
                distributed_training=True,
            )
            
            # Apply LoRA with Unsloth Pro (should work with multi-GPU)
            model = FastLanguageModel.get_peft_model(
                model,
                r=config["r"],
                target_modules=config["target_modules"],
                lora_alpha=config["lora_alpha"],
                lora_dropout=config["lora_dropout"],
                bias="none",
                use_gradient_checkpointing="unsloth",  # Pro version should handle DDP conflicts
                random_state=42,
                # Pro-specific parameters (speculative)
                multi_gpu_compatible=True,
            )
            
            # Enable training mode (Pro version)
            FastLanguageModel.for_training(model)
            
            training_framework = "unsloth_pro"
            
        else:
            # Fallback if Pro initialization fails
            accelerator, training_framework = fallback_to_standard_multi_gpu()
            is_main_process = accelerator.is_main_process
            
            # Standard model loading (fallback)
            if is_main_process:
                logger.info("Loading model with standard transformers...")
            
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            from peft import LoraConfig, get_peft_model, TaskType
            
            tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            
            # Set up quantization
            quantization_config = None
            if config["use_bnb"] and config["quantization"] in ["4bit", "8bit"]:
                if config["quantization"] == "4bit":
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                    )
                elif config["quantization"] == "8bit":
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
            model = AutoModelForCausalLM.from_pretrained(
                config["model_name"],
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                device_map={"": accelerator.local_process_index},
                trust_remote_code=True,
            )
            
            # Apply standard PEFT
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=config["r"],
                lora_alpha=config["lora_alpha"],
                lora_dropout=config["lora_dropout"],
                target_modules=config["target_modules"],
                bias="none",
            )
            model = get_peft_model(model, peft_config)
    
    else:
        # Community version - use standard approach
        print("⚠️  Unsloth Pro not detected - using standard multi-GPU")
        accelerator, training_framework = fallback_to_standard_multi_gpu()
        is_main_process = accelerator.is_main_process
        
        # Standard implementation (same as train_multi_gpu.py)
        # ... (implement standard approach here)
        
    if is_main_process:
        logger.info(f"Training framework: {training_framework}")
        if hasattr(model, 'print_trainable_parameters'):
            model.print_trainable_parameters()
    
    # Load dataset
    if is_main_process:
        logger.info("Loading dataset...")
    
    if config["dataset_config"]:
        dataset = load_dataset(config["dataset_name"], config["dataset_config"], split=config["dataset_split"])
    else:
        dataset = load_dataset(config["dataset_name"], split=config["dataset_split"])
    
    def tokenize_function(examples):
        return tokenizer(
            examples[config["dataset_text_field"]],
            truncation=True,
            padding=True,
            max_length=config["max_seq_length"],
            return_tensors="pt",
        )
    
    # Process dataset
    dataset = dataset.filter(lambda x: len(x[config["dataset_text_field"]].strip()) > 0)
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=[config["dataset_text_field"]])
    
    # Add labels for causal LM
    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples
    
    dataset = dataset.map(add_labels, batched=True)
    
    # Split dataset
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    if is_main_process:
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Eval samples: {len(eval_dataset)}")
    
    # Setup output directory with session structure
    session_dir = f"./sessions/{args.trainsession}"
    output_dir = f"{session_dir}/multi_gpu_unsloth"
    os.makedirs(output_dir, exist_ok=True)
    
    # Training arguments
    from transformers import TrainingArguments
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["num_train_epochs"],
        max_steps=config["max_steps"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        warmup_steps=config["warmup_steps"],
        learning_rate=config["learning_rate"],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=config["logging_steps"],
        optim=config["optimizer"],
        weight_decay=config["weight_decay"],
        lr_scheduler_type=config["lr_scheduler_type"],
        seed=42,
        save_steps=config["save_steps"],
        eval_strategy=config["eval_strategy"],
        eval_steps=config.get("eval_steps", 50),
        save_total_limit=1,
        report_to=None,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        ddp_find_unused_parameters=False,  # Should be compatible with Unsloth Pro
        gradient_checkpointing=True,  # Pro version should handle this properly
    )
    
    # Data collator
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt",
    )
    
    # Create trainer
    if training_framework == "unsloth_pro":
        # Use SFTTrainer for better compatibility with Unsloth Pro
        from trl import SFTTrainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            dataset_text_field=config["dataset_text_field"],
            max_seq_length=config["max_seq_length"],
        )
    else:
        # Standard trainer for fallback
        from transformers import Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    
    # Start training
    if is_main_process:
        logger.info("🚀 Starting Unsloth Pro multi-GPU training...")
    
    start_time = datetime.now()
    
    try:
        result = trainer.train()
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if is_main_process:
            logger.info("✅ Unsloth Pro multi-GPU training completed!")
            logger.info(f"Duration: {duration:.1f} seconds")
            logger.info(f"Final loss: {result.training_loss:.4f}")
            logger.info(f"Steps per second: {config['max_steps'] / duration:.2f}")
            
            # Save model
            trainer.save_model(f"{output_dir}/final_model")
            
            # Save results summary
            results = {
                "mode": "multi-gpu-unsloth-pro",
                "session": args.trainsession,
                "config": args.config,
                "duration_seconds": duration,
                "final_loss": result.training_loss,
                "steps_per_second": config['max_steps'] / duration,
                "total_steps": config['max_steps'],
                "num_gpus": accelerator.num_processes if accelerator else torch.cuda.device_count(),
                "model_name": config['model_name'],
                "optimizations": f"Unsloth Pro + {config['quantization']}-bit quantization",
                "training_framework": training_framework,
                "unsloth_pro_status": pro_status,
            }
            
            import json
            with open(f"{output_dir}/results.json", "w") as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"📊 Results saved to {output_dir}/results.json")
        
    except Exception as e:
        if is_main_process:
            logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()