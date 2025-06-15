#!/usr/bin/env python3

"""
Single GPU Training Script - Optimized with Unsloth
"""

# Import unsloth FIRST
import unsloth
from unsloth import FastLanguageModel

import os
import torch
import logging
from datetime import datetime
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

# Disable wandb and warnings
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Configuration
    config = {
        "model_name": "unsloth/llama-2-7b-bnb-4bit",
        "max_seq_length": 512,
        "max_steps": 50,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
    }
    
    # Force single GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    print("="*60)
    print("🔥 SINGLE GPU TRAINING (Unsloth Optimized)")
    print("="*60)
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
        load_in_4bit=True,  # 4-bit for memory efficiency
        trust_remote_code=True,
    )
    
    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["r"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth optimization
        random_state=42,
    )
    
    # Enable training mode
    FastLanguageModel.for_training(model)
    
    logger.info("Model loaded with Unsloth optimizations")
    model.print_trainable_parameters()
    
    # Load dataset
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
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Eval samples: {len(eval_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./single_gpu_output",
        num_train_epochs=1,
        max_steps=config["max_steps"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        warmup_steps=5,
        learning_rate=config["learning_rate"],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
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
    )
    
    # Data collator
    from transformers import DataCollatorForLanguageModeling
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