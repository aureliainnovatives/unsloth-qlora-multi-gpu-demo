#!/usr/bin/env python3

"""
Multi-GPU compatible training script without Unsloth
Uses standard transformers + PEFT for reliable multi-GPU training
"""

# Disable wandb and tokenizer warnings
import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
from peft import LoraConfig, get_peft_model, TaskType
from datetime import datetime

from config import Config
from data import create_dataloaders
from utils import setup_logging, save_config
from training_logger import create_training_logger, log_model_parameters, log_dataset_info
from enhanced_callbacks import EnhancedTrainingCallback, MultiGPUMonitoringCallback, PerformanceMonitoringCallback

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Multi-GPU Compatible QLoRA Fine-tuning")
    parser.add_argument("--force-single-gpu", action="store_true", 
                       help="Force single GPU training")
    parser.add_argument("--gpu-ids", type=str, 
                       help="Comma-separated GPU IDs to use")
    return parser.parse_args()


def setup_gpu_environment(config, args):
    """Setup GPU environment based on configuration"""
    
    available_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {available_gpus}")
    
    if available_gpus == 0:
        print("WARNING: No CUDA GPUs available. Using CPU.")
        return False
    
    if args.force_single_gpu or config.force_single_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("FORCED SINGLE GPU MODE - Using GPU 0 only")
        return False
    
    if args.gpu_ids:
        config.gpu_device_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        gpu_list = ','.join(map(str, config.gpu_device_ids))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
        print(f"Using specific GPUs: {gpu_list}")
        return len(config.gpu_device_ids) > 1
    
    if config.use_multi_gpu and available_gpus > 1:
        print(f"MULTI-GPU MODE - Using all {available_gpus} GPUs")
        return True
    else:
        print("SINGLE GPU MODE - Using GPU 0 only")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        return False


def load_model_and_tokenizer_standard(config, accelerator):
    """Load model and tokenizer using standard transformers (multi-GPU compatible)"""
    
    logger.info(f"Loading model: {config.model_name}")
    
    # Use base model name without unsloth prefix
    base_model_name = config.model_name.replace("unsloth/", "").replace("-bnb-4bit", "")
    if "llama-2-7b" in base_model_name:
        base_model_name = "meta-llama/Llama-2-7b-hf"
    
    logger.info(f"Using base model: {base_model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    if accelerator.num_processes > 1:
        # Multi-GPU: Load on each device
        logger.info("Multi-GPU: Loading model with device mapping")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
            device_map={"": accelerator.local_process_index},
            trust_remote_code=True,
        )
    else:
        # Single GPU: Load normally
        logger.info("Single GPU: Loading model normally")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
            trust_remote_code=True,
        )
    
    # Apply LoRA with standard PEFT
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias=config.bias,
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    logger.info("Model and tokenizer loaded with standard PEFT")
    return model, tokenizer


def setup_training_arguments(config, accelerator):
    """Setup training arguments"""
    
    world_size = accelerator.num_processes
    total_batch_size = config.per_device_train_batch_size * config.gradient_accumulation_steps * world_size
    
    logger.info(f"Training configuration:")
    logger.info(f"  World size: {world_size}")
    logger.info(f"  Per device batch size: {config.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation steps: {config.gradient_accumulation_steps}")
    logger.info(f"  Total batch size: {total_batch_size}")
    
    return TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=1,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        
        # Optimization
        optim="adamw_torch",  # Use standard optimizer
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        lr_scheduler_type=config.lr_scheduler_type,
        
        # Mixed precision
        fp16=config.fp16,
        bf16=config.bf16,
        
        # Logging and saving
        logging_dir=config.logging_dir,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        
        # Evaluation
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        
        # Data loading
        dataloader_pin_memory=config.dataloader_pin_memory,
        dataloader_num_workers=config.dataloader_num_workers,
        
        # Misc
        seed=config.seed,
        report_to=None,
        run_name=f"standard-qlora-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        remove_unused_columns=True,
        load_best_model_at_end=config.save_best_model,
        metric_for_best_model="eval_loss" if config.eval_strategy != "no" else None,
        greater_is_better=False,
    )


def main():
    # Parse arguments
    args = parse_args()
    
    # Initialize config
    config = Config()
    
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
    print(f"Using Standard Transformers + PEFT (Multi-GPU Compatible)")
    print(f"{'='*60}\n")
    
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=None,
    )
    
    # Setup logging
    setup_logging(accelerator)
    
    # Create enhanced training logger
    training_logger = create_training_logger(config, accelerator)
    
    # Set seed
    set_seed(config.seed)
    
    # Save config
    if accelerator.is_main_process:
        save_config(config, config.output_dir)
    
    logger.info("Starting Standard QLoRA Multi-GPU Fine-tuning")
    logger.info(f"Using {accelerator.num_processes} GPUs")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer_standard(config, accelerator)
    
    # Log model information
    log_model_parameters(training_logger, model)
    
    # Prepare datasets
    data_processor = create_dataloaders(config, accelerator)
    data_processor.load_tokenizer(model)
    train_dataset, eval_dataset = data_processor.load_dataset()
    train_dataset, eval_dataset = data_processor.prepare_datasets()
    
    # Log dataset information
    log_dataset_info(training_logger, train_dataset, eval_dataset)
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt",
    )
    
    # Setup training arguments
    training_args = setup_training_arguments(config, accelerator)
    
    # Create callbacks
    callbacks = [
        EnhancedTrainingCallback(training_logger),
        MultiGPUMonitoringCallback(training_logger),
        PerformanceMonitoringCallback(training_logger),
    ]
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
        callbacks=callbacks,
    )
    
    # Start training
    logger.info("Starting training...")
    
    try:
        train_result = trainer.train()
        
        if accelerator.is_main_process:
            logger.info("Training completed successfully!")
            logger.info(f"Final loss: {train_result.training_loss:.4f}")
            
            # Save final model
            trainer.save_model(os.path.join(config.output_dir, "final_model"))
            
            logger.info(f"Final model saved to: {os.path.join(config.output_dir, 'final_model')}")
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()