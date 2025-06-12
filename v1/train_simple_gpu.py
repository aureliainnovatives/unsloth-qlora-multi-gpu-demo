#!/usr/bin/env python3

"""
Simplified GPU training without bitsandbytes/unsloth dependencies
"""

import os
import logging
import torch
import argparse
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from datetime import datetime

from config import Config

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Simple GPU QLoRA Fine-tuning")
    parser.add_argument("--force-single-gpu", action="store_true", 
                       help="Force single GPU training")
    parser.add_argument("--gpu-ids", type=str, 
                       help="Comma-separated GPU IDs to use")
    return parser.parse_args()


def setup_gpu_environment(config, args):
    """Setup GPU environment"""
    
    available_gpus = torch.cuda.device_count()
    logger.info(f"Available GPUs: {available_gpus}")
    
    if available_gpus == 0:
        logger.error("No CUDA GPUs available!")
        return False
    
    if args.force_single_gpu or config.force_single_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        logger.info("FORCED SINGLE GPU MODE - Using GPU 0 only")
        return False
    
    if args.gpu_ids:
        config.gpu_device_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        gpu_list = ','.join(map(str, config.gpu_device_ids))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
        logger.info(f"Using specific GPUs: {gpu_list}")
        return len(config.gpu_device_ids) > 1
    
    if config.use_multi_gpu and available_gpus > 1:
        logger.info(f"MULTI-GPU MODE - Using all {available_gpus} GPUs")
        return True
    else:
        logger.info("SINGLE GPU MODE - Using GPU 0 only")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        return False


def load_model_and_tokenizer(config):
    """Load model and tokenizer without unsloth"""
    
    # Use a smaller model that works without 4-bit quantization
    model_name = "microsoft/DialoGPT-medium"  # or "gpt2" for testing
    
    logger.info(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model in fp16 (no quantization for now)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if config.fp16 else torch.float32,
        device_map="auto" if torch.cuda.device_count() > 1 else None,
    )
    
    # Apply LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["c_attn", "c_proj"],  # DialoGPT specific
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def prepare_dataset(tokenizer, config):
    """Prepare dataset"""
    
    logger.info("Loading dataset...")
    
    # Use a simple text dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1000]")  # Small subset
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=config.max_seq_length,
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    # Split train/eval
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    
    return split_dataset["train"], split_dataset["test"]


def main():
    # Parse arguments
    args = parse_args()
    
    # Initialize config
    config = Config()
    config.max_steps = 50  # Short for testing
    config.per_device_train_batch_size = 2
    config.gradient_accumulation_steps = 2
    config.learning_rate = 5e-4
    config.logging_steps = 5
    config.eval_steps = 20
    config.save_steps = 25
    
    if args.force_single_gpu:
        config.force_single_gpu = True
        config.use_multi_gpu = False
    
    # Setup GPU environment
    is_multi_gpu = setup_gpu_environment(config, args)
    
    # Print GPU configuration
    print(f"\n{'='*60}")
    print("GPU CONFIGURATION")
    print(f"{'='*60}")
    print(f"Mode: {'MULTI-GPU' if is_multi_gpu else 'SINGLE-GPU'}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"{'='*60}\n")
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Set seed
    set_seed(config.seed)
    
    logger.info("Starting Simple GPU Fine-tuning")
    logger.info(f"Using {accelerator.num_processes} GPUs")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Prepare dataset
    train_dataset, eval_dataset = prepare_dataset(tokenizer, config)
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Eval samples: {len(eval_dataset)}")
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./simple_gpu_output",
        num_train_epochs=1,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        fp16=config.fp16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_total_limit=2,
        remove_unused_columns=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=0,
        optim="adamw_torch",
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
    logger.info("🚀 Starting training...")
    
    try:
        result = trainer.train()
        
        if accelerator.is_main_process:
            logger.info("✅ Training completed!")
            logger.info(f"Final loss: {result.training_loss:.4f}")
            
            # Run evaluation
            eval_result = trainer.evaluate()
            logger.info(f"Eval loss: {eval_result['eval_loss']:.4f}")
            
            # Save model
            trainer.save_model("./simple_gpu_output/final_model")
            
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()