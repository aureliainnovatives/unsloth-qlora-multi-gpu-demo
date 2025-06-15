#!/usr/bin/env python3

"""
Single GPU Training Script - Optimized with Unsloth
Uses training_configs.py for configuration management
"""

# Import unsloth FIRST
import unsloth
from unsloth import FastLanguageModel

import os
import torch
import logging
from datetime import datetime
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer
from datasets import load_dataset
from training_configs import get_config, parse_training_args

# Disable wandb and warnings
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Parse arguments and get configuration
    args = parse_training_args()
    config = get_config(args.config)
    
    # Force single GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    print("="*60)
    print("🔥 SINGLE GPU TRAINING (Unsloth Optimized)")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Model: {config['model_name']}")
    print(f"Max steps: {config['max_steps']}")
    print(f"Batch size: {config['per_device_train_batch_size']}")
    print(f"Using GPU 0 only")
    print("="*60)
    
    # Load model with Unsloth (4-bit quantization)
    logger.info("Loading model with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        dtype=None,
        load_in_4bit=(config["quantization"] == "4bit"),
        load_in_8bit=(config["quantization"] == "8bit"),
        trust_remote_code=True,
    )
    
    # Apply LoRA - Check if model supports Unsloth optimizations
    if "llama" in config["model_name"].lower() or "mistral" in config["model_name"].lower():
        # Use Unsloth for supported models
        model = FastLanguageModel.get_peft_model(
            model,
            r=config["r"],
            target_modules=config["target_modules"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
    else:
        # Use standard PEFT for other models (like DialoGPT)
        from peft import LoraConfig, get_peft_model, TaskType
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
    
    # Enable training mode (only for Unsloth models)
    if "llama" in config["model_name"].lower() or "mistral" in config["model_name"].lower():
        FastLanguageModel.for_training(model)
    
    logger.info("Model loaded with Unsloth optimizations")
    model.print_trainable_parameters()
    
    # Load dataset
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
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Eval samples: {len(eval_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./single_gpu_output",
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
    )
    
    # Data collator
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt",
    )
    
    # Create trainer - Use SFTTrainer for better compatibility
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
    
    # Start training
    logger.info("🚀 Starting single GPU training...")
    start_time = datetime.now()
    
    try:
        result = trainer.train()
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("✅ Single GPU training completed!")
        logger.info(f"Duration: {duration:.1f} seconds")
        logger.info(f"Final loss: {result.training_loss:.4f}")
        logger.info(f"Steps per second: {config['max_steps'] / duration:.2f}")
        
        # Save model
        trainer.save_model("./single_gpu_output/final_model")
        
        # Save results summary
        results = {
            "mode": "single-gpu",
            "duration_seconds": duration,
            "final_loss": result.training_loss,
            "steps_per_second": config['max_steps'] / duration,
            "total_steps": config['max_steps'],
            "model_name": config['model_name'],
            "optimizations": "Unsloth + 4-bit quantization",
        }
        
        import json
        with open("./single_gpu_output/results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info("📊 Results saved to single_gpu_output/results.json")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()