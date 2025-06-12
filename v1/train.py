#!/usr/bin/env python3

# Import unsloth FIRST (before transformers/peft)
import unsloth
from unsloth import FastLanguageModel

import os
import logging
import torch
import argparse
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb
from datetime import datetime

from config import Config
from data import create_dataloaders
from utils import (
    setup_logging,
    save_config,
    EvaluationCallback,
    compute_metrics,
    print_trainable_parameters,
)
from training_logger import (
    create_training_logger,
    log_model_parameters,
    log_dataset_info,
    MultiGPUTrainingLogger
)
from enhanced_callbacks import (
    EnhancedTrainingCallback,
    MultiGPUMonitoringCallback,
    PerformanceMonitoringCallback
)

logger = get_logger(__name__)


def load_model_and_tokenizer(config):
    """Load model and tokenizer using Unsloth"""
    logger.info(f"Loading model: {config.model_name}")
    
    # Load model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
        trust_remote_code=True,
    )
    
    # Apply LoRA using Unsloth
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.r,
        target_modules=config.target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.bias,
        use_gradient_checkpointing="unsloth",  # Use Unsloth's optimized gradient checkpointing
        random_state=config.seed,
    )
    
    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Enable training mode
    model.train()
    
    logger.info("Model and tokenizer loaded successfully")
    print_trainable_parameters(model)
    
    return model, tokenizer


def setup_training_arguments(config, accelerator):
    """Setup Hugging Face training arguments"""
    
    # Calculate total batch size
    world_size = accelerator.num_processes
    total_batch_size = (
        config.per_device_train_batch_size * 
        config.gradient_accumulation_steps * 
        world_size
    )
    
    logger.info(f"Training configuration:")
    logger.info(f"  World size: {world_size}")
    logger.info(f"  Per device batch size: {config.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation steps: {config.gradient_accumulation_steps}")
    logger.info(f"  Total batch size: {total_batch_size}")
    
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=1,  # We use max_steps instead
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        
        # Optimization
        optim=config.optim,
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
        report_to=["wandb"] if wandb.run else None,
        run_name=f"qwen-7b-qlora-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        
        # Remove unused columns automatically
        remove_unused_columns=True,
        
        # Don't load best model at end (we'll handle this manually)
        load_best_model_at_end=config.save_best_model,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    return training_args


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="QLoRA Multi-GPU Fine-tuning")
    parser.add_argument("--force-single-gpu", action="store_true", 
                       help="Force single GPU training even if multiple GPUs available")
    parser.add_argument("--gpu-ids", type=str, 
                       help="Comma-separated GPU IDs to use (e.g., '0,1')")
    parser.add_argument("--config-file", type=str,
                       help="Path to custom config file")
    return parser.parse_args()


def setup_gpu_environment(config, args):
    """Setup GPU environment based on configuration"""
    
    # Check available GPUs
    available_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {available_gpus}")  # Use print instead of logger before accelerator init
    
    if available_gpus == 0:
        print("WARNING: No CUDA GPUs available. Using CPU.")
        return False
    
    # Handle force single GPU
    if args.force_single_gpu or config.force_single_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("FORCED SINGLE GPU MODE - Using GPU 0 only")
        return False
    
    # Handle specific GPU IDs
    if args.gpu_ids:
        config.gpu_device_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    
    if config.gpu_device_ids:
        gpu_list = ','.join(map(str, config.gpu_device_ids))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
        print(f"Using specific GPUs: {gpu_list}")
        return len(config.gpu_device_ids) > 1
    
    # Use all available GPUs for multi-GPU if enabled
    if config.use_multi_gpu and available_gpus > 1:
        print(f"MULTI-GPU MODE - Using all {available_gpus} GPUs")
        return True
    else:
        print("SINGLE GPU MODE - Using GPU 0 only")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        return False


def main():
    # Parse arguments
    args = parse_args()
    
    # Initialize config
    config = Config()
    
    # Override config with command line args
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
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="wandb" if os.getenv("WANDB_API_KEY") else None,
    )
    
    # Setup logging
    setup_logging(accelerator)
    
    # Create enhanced training logger
    training_logger = create_training_logger(config, accelerator)
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Save config
    if accelerator.is_main_process:
        save_config(config, config.output_dir)
        
        # Initialize wandb if API key is available
        if os.getenv("WANDB_API_KEY"):
            wandb.init(
                project="qwen-7b-qlora-multigpu",
                name=f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config=config.__dict__,
            )
    
    logger.info("Starting QLoRA Multi-GPU Fine-tuning")
    logger.info(f"Using {accelerator.num_processes} GPUs")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
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
    data_collator = data_processor.create_data_collator()
    
    # Setup training arguments
    training_args = setup_training_arguments(config, accelerator)
    
    # Create callbacks
    callbacks = []
    if config.save_best_model:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
    
    callbacks.append(EvaluationCallback())
    
    # Add enhanced logging callbacks
    callbacks.append(EnhancedTrainingCallback(training_logger))
    callbacks.append(MultiGPUMonitoringCallback(training_logger))
    callbacks.append(PerformanceMonitoringCallback(training_logger))
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    
    # Start training
    logger.info("Starting training...")
    
    # Resume from checkpoint if specified
    checkpoint = None
    if config.resume_from_checkpoint:
        checkpoint = config.resume_from_checkpoint
        logger.info(f"Resuming from checkpoint: {checkpoint}")
    
    # Train
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    # Save final model
    if accelerator.is_main_process:
        logger.info("Training completed. Saving final model...")
        
        # Save with Unsloth for optimized inference
        model.save_pretrained(os.path.join(config.output_dir, "final_model"))
        tokenizer.save_pretrained(os.path.join(config.output_dir, "final_model"))
        
        # Save training metrics
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        
        # Final evaluation
        logger.info("Running final evaluation...")
        eval_result = trainer.evaluate()
        trainer.log_metrics("eval", eval_result)
        trainer.save_metrics("eval", eval_result)
        
        logger.info("Training and evaluation completed successfully!")
        logger.info(f"Final model saved to: {os.path.join(config.output_dir, 'final_model')}")
        
        # Close wandb
        if wandb.run:
            wandb.finish()


if __name__ == "__main__":
    main()