"""
Training Script for Bug Report Classifier
Optimized for Kaggle environment with GPU
"""

import os
import json
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import argparse

def load_data(train_path, val_path):
    """Load training and validation data from JSONL files"""
    print("📂 Loading datasets...")
    
    # Load JSONL files
    train_data = []
    with open(train_path, 'r') as f:
        for line in f:
            train_data.append(json.loads(line))
    
    val_data = []
    with open(val_path, 'r') as f:
        for line in f:
            val_data.append(json.loads(line))
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    print(f"   Train: {len(train_dataset)} examples")
    print(f"   Val: {len(val_dataset)} examples")
    
    return train_dataset, val_dataset

def format_prompt(example):
    """Format example into a prompt for the model"""
    return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Output:
{example['output']}"""

def tokenize_function(example, tokenizer, max_length=512):
    """Tokenize a single example"""
    # Format prompt
    prompt = format_prompt(example)
    
    # Tokenize (don't use return_tensors for dataset mapping)
    tokenized = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding='max_length',
    )
    
    # For causal LM, labels are the same as input_ids
    tokenized['labels'] = tokenized['input_ids'].copy()
    
    return tokenized

def setup_model(model_name, lora_r=8, lora_alpha=32, use_4bit=True):
    """Load and setup model with LoRA"""
    print(f"\n🤖 Loading model: {model_name}")
    print(f"   LoRA config: r={lora_r}, α={lora_alpha}, α/r={lora_alpha/lora_r:.1f}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model with quantization if specified
    if use_4bit:
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
        )
    
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Setup LoRA
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def train(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    train_path="/kaggle/input/data-split/train_improved.jsonl",
    val_path="/kaggle/input/data-split/val.jsonl",
    output_dir="/kaggle/working/outputs",
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-4,
    max_length=512,
    lora_r=8,
    lora_alpha=32,
):
    """Main training function"""
    
    print("="*60)
    print("🎮 Game Bug Report Classifier Training")
    print("="*60)
    
    # Load data
    train_dataset, val_dataset = load_data(train_path, val_path)
    
    # Setup model
    model, tokenizer = setup_model(model_name, lora_r=lora_r, lora_alpha=lora_alpha, use_4bit=True)
    
    # Tokenize datasets
    print("\n🔤 Tokenizing datasets...")
    
    def tokenize_example(example):
        return tokenize_function(example, tokenizer, max_length)
    
    train_dataset = train_dataset.map(
        tokenize_example,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing training data"
    )
    
    val_dataset = val_dataset.map(
        tokenize_example,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation data"
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        fp16=True,
        save_strategy="steps",
        save_steps=200,
        eval_strategy="steps",
        eval_steps=200,
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=False,
        report_to=["tensorboard"],
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\n🚀 Starting training...")
    trainer.train()
    
    # Save final model
    print("\n💾 Saving final model...")
    final_output_dir = os.path.join(output_dir, "final_model")
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    print(f"\n✅ Training complete! Model saved to: {final_output_dir}")
    
    return trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--train_path", type=str, default="/kaggle/input/data-split/train_improved.jsonl")
    parser.add_argument("--val_path", type=str, default="/kaggle/input/data-split/val.jsonl")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/outputs")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank (number of low-rank dimensions)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (scaling factor)")
    
    args = parser.parse_args()
    
    train(
        model_name=args.model_name,
        train_path=args.train_path,
        val_path=args.val_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
