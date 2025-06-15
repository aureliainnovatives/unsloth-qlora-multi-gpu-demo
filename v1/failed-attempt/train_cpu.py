#!/usr/bin/env python3

"""
CPU-only training script for testing without GPU dependencies
"""

import os
import logging
import torch
import argparse
from datetime import datetime

# Use CPU-compatible imports only
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

from config import Config

# Simple CPU logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_simple_model_and_tokenizer(config):
    """Load a simple model for CPU testing"""
    
    # Use a small model for CPU testing
    model_name = "microsoft/DialoGPT-small"  # Small model for CPU
    
    logger.info(f"Loading CPU model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer


def prepare_simple_dataset(tokenizer, config):
    """Prepare a simple dataset for CPU testing"""
    
    # Use a small dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:100]")  # Only 100 samples
    
    def tokenize_function(examples):
        # Simple tokenization
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=128,  # Short sequences for CPU
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    # Split into train/eval
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    
    return split_dataset["train"], split_dataset["test"]


def main():
    parser = argparse.ArgumentParser(description="CPU-only training test")
    parser.add_argument("--steps", type=int, default=10, help="Number of training steps")
    args = parser.parse_args()
    
    # Initialize config
    config = Config()
    config.max_steps = args.steps
    config.per_device_train_batch_size = 1
    config.learning_rate = 5e-5
    config.logging_steps = 1
    config.save_steps = 50
    config.eval_steps = 5
    
    logger.info("="*60)
    logger.info("🖥️  CPU-ONLY TRAINING TEST")
    logger.info("="*60)
    logger.info(f"Steps: {config.max_steps}")
    logger.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Load model and tokenizer
    model, tokenizer = load_simple_model_and_tokenizer(config)
    
    # Prepare dataset
    train_dataset, eval_dataset = prepare_simple_dataset(tokenizer, config)
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Eval samples: {len(eval_dataset)}")
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./cpu_test_output",
        num_train_epochs=1,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_train_batch_size,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_total_limit=1,
        remove_unused_columns=True,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
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
    logger.info("🚀 Starting CPU training...")
    
    try:
        result = trainer.train()
        logger.info("✅ Training completed successfully!")
        logger.info(f"Final loss: {result.training_loss:.4f}")
        
        # Run evaluation
        eval_result = trainer.evaluate()
        logger.info(f"Eval loss: {eval_result['eval_loss']:.4f}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()