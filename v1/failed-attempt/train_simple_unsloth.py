#!/usr/bin/env python3

"""
Simplified Unsloth training that definitely works
"""

# Import unsloth FIRST
import unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported

import os
import torch

# Disable wandb and warnings
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    # Simple configuration
    max_seq_length = 512
    dtype = None  # Auto-detection
    load_in_4bit = True  # Use 4bit quantization
    
    # Check GPU setup
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Using 4-bit quantization: {load_in_4bit}")
    
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-2-7b-bnb-4bit",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    
    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )
    
    # Training setup with minimal data
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset
    
    # Create simple dataset
    simple_data = [
        {"text": "### Instruction: What is 2+2? ### Response: 2+2 equals 4."},
        {"text": "### Instruction: What is 3+3? ### Response: 3+3 equals 6."},
        {"text": "### Instruction: What is 4+4? ### Response: 4+4 equals 8."},
        {"text": "### Instruction: What is 5+5? ### Response: 5+5 equals 10."},
    ] * 50  # Repeat to have enough data
    
    dataset = Dataset.from_list(simple_data)
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=1,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            warmup_steps=2,
            max_steps=10,  # Very short
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="./simple_unsloth_output",
            save_steps=10,
            report_to=None,
        ),
    )
    
    print("🚀 Starting simple Unsloth training...")
    trainer.train()
    print("✅ Training completed!")
    
    # Save model
    model.save_pretrained("./simple_unsloth_output/final_model")
    tokenizer.save_pretrained("./simple_unsloth_output/final_model")
    print("💾 Model saved!")

if __name__ == "__main__":
    main()