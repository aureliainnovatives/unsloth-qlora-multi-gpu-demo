#!/usr/bin/env python3

"""
Unsloth native multi-GPU training approach
Using Unsloth's recommended method for multi-GPU
"""

# Import unsloth FIRST
import unsloth
from unsloth import FastLanguageModel

import os
import logging
import torch
import argparse
from datetime import datetime

# Disable wandb and warnings
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import TrainingArguments, Trainer
from trl import SFTTrainer  # Unsloth recommends SFTTrainer for multi-GPU
from datasets import load_dataset

from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Unsloth Native Multi-GPU Training")
    parser.add_argument("--force-single-gpu", action="store_true", 
                       help="Force single GPU training")
    return parser.parse_args()


def setup_gpu_environment(args):
    """Setup GPU environment"""
    
    available_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {available_gpus}")
    
    if available_gpus == 0:
        print("WARNING: No CUDA GPUs available.")
        return False
    
    if args.force_single_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("FORCED SINGLE GPU MODE - Using GPU 0 only")
        return False
    
    if available_gpus > 1:
        print(f"MULTI-GPU MODE - Using all {available_gpus} GPUs")
        return True
    else:
        print("SINGLE GPU MODE - Using GPU 0 only")
        return False


def load_model_and_tokenizer_unsloth(config, is_multi_gpu):
    """Load model using Unsloth's recommended approach"""
    
    logger.info(f"Loading model: {config.model_name}")
    
    # Unsloth native loading
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=not is_multi_gpu,  # 4-bit for single GPU, full precision for multi-GPU
        trust_remote_code=True,
    )
    
    # Apply LoRA with Unsloth
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.r,
        target_modules=config.target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.bias,
        use_gradient_checkpointing="unsloth" if not is_multi_gpu else False,
        random_state=config.seed,
    )
    
    # Enable for training
    FastLanguageModel.for_training(model)
    
    logger.info("Model loaded with Unsloth native approach")
    return model, tokenizer


def format_prompts(examples):
    """Format prompts for instruction tuning"""
    texts = []
    
    # Check what fields are available
    if "text" in examples:
        # If text field already exists, use it directly
        return {"text": examples["text"]}
    elif "conversations" in examples:
        # Format conversations
        for conversation in examples["conversations"]:
            if isinstance(conversation, list) and len(conversation) > 0:
                text = ""
                for turn in conversation:
                    if "from" in turn and "value" in turn:
                        if turn["from"] == "human":
                            text += f"### Instruction:\n{turn['value']}\n\n"
                        elif turn["from"] == "gpt":
                            text += f"### Response:\n{turn['value']}\n\n"
                texts.append(text)
            else:
                texts.append(str(conversation))
    else:
        # Fallback: use any available text field
        for key in examples.keys():
            if isinstance(examples[key], list) and len(examples[key]) > 0:
                if isinstance(examples[key][0], str):
                    texts = examples[key]
                    break
        
        if not texts:
            # Last resort: convert first field to strings
            first_key = list(examples.keys())[0]
            texts = [str(item) for item in examples[first_key]]
    
    return {"text": texts}


def main():
    # Parse arguments
    args = parse_args()
    
    # Initialize config
    config = Config()
    config.max_steps = 50  # Keep short for testing
    
    # Setup GPU environment
    is_multi_gpu = setup_gpu_environment(args)
    
    # Print configuration
    print(f"\n{'='*60}")
    print("UNSLOTH NATIVE GPU CONFIGURATION")
    print(f"{'='*60}")
    print(f"Mode: {'MULTI-GPU' if is_multi_gpu else 'SINGLE-GPU'}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Using Unsloth Native Approach")
    print(f"{'='*60}\n")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer_unsloth(config, is_multi_gpu)
    
    # Load and prepare dataset
    logger.info("Loading dataset...")
    dataset = load_dataset("timdettmers/openassistant-guanaco", split="train[:1000]")
    
    # Debug: Check dataset structure
    logger.info(f"Dataset columns: {dataset.column_names}")
    if len(dataset) > 0:
        logger.info(f"First example keys: {list(dataset[0].keys())}")
    
    # Format dataset for Unsloth
    dataset = dataset.map(format_prompts, batched=True)
    
    # Split for evaluation
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Eval samples: {len(eval_dataset)}")
    
    # Training arguments for Unsloth
    training_args = TrainingArguments(
        output_dir="./unsloth_native_output",
        num_train_epochs=1,
        max_steps=config.max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        
        # Optimization
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        
        # Saving and evaluation
        save_steps=50,
        eval_strategy="no",  # Disable evaluation for stability
        save_total_limit=1,
        
        # Disable wandb
        report_to=None,
        
        # Multi-GPU settings
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    
    # Use SFTTrainer (Supervised Fine-tuning Trainer) recommended by Unsloth
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        dataset_num_proc=2,
        packing=False,  # Disable packing for multi-GPU stability
        args=training_args,
    )
    
    # Start training
    logger.info("🚀 Starting Unsloth native training...")
    
    try:
        # Show model info
        trainer.model.print_trainable_parameters()
        
        # Train
        train_result = trainer.train()
        
        logger.info("✅ Training completed successfully!")
        logger.info(f"Final loss: {train_result.training_loss:.4f}")
        
        # Save model
        trainer.save_model(f"./unsloth_native_output/final_model")
        logger.info("Model saved successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()