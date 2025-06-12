#!/usr/bin/env python3

"""
Quick test script to check if models can be loaded
"""

import torch
from unsloth import FastLanguageModel

def test_model(model_name):
    """Test if a model can be loaded"""
    print(f"\n🧪 Testing model: {model_name}")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=512,  # Small for testing
            dtype=None,
            load_in_4bit=True,
        )
        print(f"✅ SUCCESS: {model_name} loaded successfully!")
        print(f"   Model type: {type(model)}")
        print(f"   Tokenizer vocab size: {len(tokenizer)}")
        return True
    except Exception as e:
        print(f"❌ FAILED: {model_name}")
        print(f"   Error: {str(e)[:100]}...")
        return False

def main():
    print("🔍 Testing Unsloth Model Availability")
    print("="*50)
    
    # List of known working Unsloth models
    models_to_test = [
        "unsloth/llama-2-7b-bnb-4bit",
        "unsloth/llama-2-7b-chat-bnb-4bit", 
        "unsloth/mistral-7b-bnb-4bit",
        "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
        "unsloth/codellama-7b-bnb-4bit",
        "unsloth/tinyllama-bnb-4bit",
        "unsloth/qwen-7b-qlora",  # Original one that failed
    ]
    
    working_models = []
    
    for model_name in models_to_test:
        if test_model(model_name):
            working_models.append(model_name)
    
    print(f"\n📊 RESULTS:")
    print(f"Working models: {len(working_models)}")
    print(f"Failed models: {len(models_to_test) - len(working_models)}")
    
    if working_models:
        print(f"\n✅ RECOMMENDED MODEL:")
        print(f"   {working_models[0]}")
        print(f"\n📝 Update config.py with:")
        print(f'   model_name: str = "{working_models[0]}"')
    else:
        print(f"\n❌ No models working! Check internet connection or try different models.")

if __name__ == "__main__":
    main()