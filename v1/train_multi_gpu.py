#!/usr/bin/env python3

"""
Multi-GPU Training Script - Standard Transformers + PEFT
"""

import os
import torch
import logging
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from accelerate import Accelerator

# Disable wandb and warnings
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Configuration
    config = {
        "model_name": "meta-llama/Llama-2-7b-hf",
        "max_seq_length": 512,
        "max_steps": 50,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
    }
    
    # Initialize accelerator for multi-GPU
    accelerator = Accelerator()
    
    print("="*60)
    print("🚀 MULTI-GPU TRAINING (Standard Transformers)")
    print("="*60)
    print(f"Model: {config['model_name']}")
    print(f"Max steps: {config['max_steps']}")
    print(f"Batch size per device: {config['per_device_train_batch_size']}")
    print(f"Number of GPUs: {accelerator.num_processes}")
    print(f"Current GPU: {accelerator.local_process_index}")
    print(f"Total batch size: {config['per_device_train_batch_size'] * config['gradient_accumulation_steps'] * accelerator.num_processes}")
    print("="*60)
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model (full precision for multi-GPU)
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map={"": accelerator.local_process_index},
        trust_remote_code=True,
    )
    
    # Apply LoRA with PEFT
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config["r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    
    model = get_peft_model(model, peft_config)
    
    if accelerator.is_main_process:
        logger.info("Model loaded with standard PEFT")
        model.print_trainable_parameters()
    
    # Load dataset
    if accelerator.is_main_process:
        logger.info("Loading dataset...")
    
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1000]")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=config["max_seq_length"],
            return_tensors="pt",
        )
    
    # Process dataset
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Add labels for causal LM
    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples
    
    dataset = dataset.map(add_labels, batched=True)
    
    # Split dataset
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    if accelerator.is_main_process:
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Eval samples: {len(eval_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./multi_gpu_output",
        num_train_epochs=1,
        max_steps=config["max_steps"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        warmup_steps=5,
        learning_rate=config["learning_rate"],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_torch",  # Standard optimizer for multi-GPU
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        save_steps=50,
        eval_strategy="no",  # Disable evaluation for simplicity
        save_total_limit=1,
        report_to=None,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        ddp_find_unused_parameters=False,  # Optimize DDP
        gradient_checkpointing=False,  # Disable for stability
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt",
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Start training
    if accelerator.is_main_process:
        logger.info("🚀 Starting multi-GPU training...")
    
    start_time = datetime.now()
    
    try:
        result = trainer.train()
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if accelerator.is_main_process:
            logger.info("✅ Multi-GPU training completed!")
            logger.info(f"Duration: {duration:.1f} seconds")
            logger.info(f"Final loss: {result.training_loss:.4f}")
            logger.info(f"Steps per second: {config['max_steps'] / duration:.2f}")
            
            # Save model
            trainer.save_model("./multi_gpu_output/final_model")
            
            # Save results summary
            results = {
                "mode": "multi-gpu",
                "duration_seconds": duration,
                "final_loss": result.training_loss,
                "steps_per_second": config['max_steps'] / duration,
                "total_steps": config['max_steps'],
                "num_gpus": accelerator.num_processes,
                "model_name": config['model_name'],
                "optimizations": "Standard transformers + PEFT",
            }
            
            import json
            with open("./multi_gpu_output/results.json", "w") as f:
                json.dump(results, f, indent=2)
            
            logger.info("📊 Results saved to multi_gpu_output/results.json")
        
    except Exception as e:
        if accelerator.is_main_process:
            logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()