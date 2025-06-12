import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class DatasetProcessor:
    def __init__(self, config):
        self.config = config
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None
    
    def load_tokenizer(self, model):
        """Load and configure tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        logger.info(f"Loaded tokenizer for {self.config.model_name}")
        logger.info(f"Vocab size: {len(self.tokenizer)}")
        logger.info(f"Pad token: {self.tokenizer.pad_token}")
        
        return self.tokenizer
    
    def load_dataset(self):
        """Load and preprocess dataset"""
        logger.info(f"Loading dataset: {self.config.dataset_name}")
        
        # Load dataset
        dataset = load_dataset(self.config.dataset_name)
        
        # Get train split
        if self.config.dataset_split in dataset:
            train_data = dataset[self.config.dataset_split]
        else:
            train_data = dataset["train"]
        
        # Limit samples if specified
        if self.config.max_samples:
            train_data = train_data.select(range(min(self.config.max_samples, len(train_data))))
        
        # Split train/eval (90/10 split)
        train_eval_split = train_data.train_test_split(test_size=0.1, seed=self.config.seed)
        self.train_dataset = train_eval_split["train"]
        self.eval_dataset = train_eval_split["test"]
        
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Eval samples: {len(self.eval_dataset)}")
        
        return self.train_dataset, self.eval_dataset
    
    def tokenize_function(self, examples):
        """Tokenize examples"""
        # Get text field
        texts = examples[self.config.dataset_text_field]
        
        # Filter out empty texts
        texts = [text for text in texts if text and len(text.strip()) > 0]
        
        if not texts:
            # Return empty batch if no valid texts
            return {"input_ids": [], "attention_mask": [], "labels": []}
        
        # Tokenize with padding and truncation
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=True,  # Enable padding
            max_length=self.config.max_seq_length,
            return_overflowing_tokens=False,
            return_tensors=None,  # Keep as lists for now
        )
        
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
        
        return tokenized
    
    def prepare_datasets(self):
        """Tokenize and prepare datasets"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_tokenizer first.")
        
        if self.train_dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset first.")
        
        logger.info("Tokenizing datasets...")
        
        # Tokenize datasets
        self.train_dataset = self.train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=self.train_dataset.column_names,
            desc="Tokenizing train dataset",
        )
        
        self.eval_dataset = self.eval_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=self.eval_dataset.column_names,
            desc="Tokenizing eval dataset",
        )
        
        # Set format for PyTorch
        self.train_dataset.set_format(type="torch")
        self.eval_dataset.set_format(type="torch")
        
        logger.info("Dataset tokenization completed")
        
        return self.train_dataset, self.eval_dataset
    
    def create_data_collator(self):
        """Create data collator for dynamic padding"""
        from transformers import DataCollatorForLanguageModeling
        
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
            pad_to_multiple_of=8,  # For efficiency on modern hardware
            return_tensors="pt",  # Ensure PyTorch tensors
        )


def create_dataloaders(config, accelerator=None):
    """Create train and eval dataloaders"""
    processor = DatasetProcessor(config)
    
    # This will be called after model loading in train.py
    # since we need the model to properly initialize the tokenizer
    return processor


class ChatTemplate:
    """Handle different chat templates for instruction datasets"""
    
    @staticmethod
    def format_alpaca_style(instruction: str, input_text: str = "", output: str = "") -> str:
        """Format data in Alpaca style"""
        if input_text:
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    
    @staticmethod
    def format_openassistant_style(conversations: List[Dict]) -> str:
        """Format OpenAssistant style conversations"""
        formatted = ""
        for conv in conversations:
            role = conv.get("from", "")
            content = conv.get("value", "")
            
            if role == "human":
                formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "gpt":
                formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        
        return formatted
    
    @staticmethod
    def auto_detect_and_format(example: Dict) -> str:
        """Auto-detect format and apply appropriate template"""
        if "conversations" in example:
            return ChatTemplate.format_openassistant_style(example["conversations"])
        elif "instruction" in example:
            return ChatTemplate.format_alpaca_style(
                example["instruction"],
                example.get("input", ""),
                example.get("output", "")
            )
        elif "text" in example:
            return example["text"]
        else:
            # Fallback: use the first string field found
            for key, value in example.items():
                if isinstance(value, str):
                    return value
            raise ValueError(f"Could not find text field in example: {example.keys()}")