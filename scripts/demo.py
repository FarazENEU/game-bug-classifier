"""
Interactive Demo for Bug Report Classifier
Run this to test the model with custom inputs
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse


def load_model(model_path, base_model_name):
    """Load the fine-tuned model"""
    print("🔄 Loading model...")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    print("✅ Model loaded!\n")
    return model, tokenizer


def classify_bug(model, tokenizer, title, description):
    """Classify a bug report"""
    input_text = f"Title: {title}\n\nDescription: {description}"
    
    prompt = f"""### Instruction:
Classify this bug report and provide a structured analysis.

### Input:
{input_text}

### Output:
"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    print("🤖 Analyzing bug report...\n")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    return generated_text.strip()


def interactive_demo(model, tokenizer):
    """Run interactive demo"""
    print("="*60)
    print("🐛 GAME BUG CLASSIFIER - INTERACTIVE DEMO")
    print("="*60)
    print("Enter bug reports to get instant classification!")
    print("Type 'quit' or 'exit' to stop.\n")
    
    # Example bugs for quick testing
    examples = [
        {
            "title": "Game crashes when opening inventory",
            "description": "Whenever I press 'I' to open inventory, the game immediately crashes to desktop. No error message shown. Happens 100% of the time."
        },
        {
            "title": "Character model is invisible",
            "description": "My character is completely invisible in third-person view. I can still move and interact, but can't see the model."
        },
        {
            "title": "Audio stuttering in main menu",
            "description": "Background music in the main menu stutters every few seconds. Doesn't happen in-game, only in menus."
        }
    ]
    
    print("Quick Test Examples (type number to use):")
    for i, ex in enumerate(examples, 1):
        print(f"{i}. {ex['title']}")
    print("Or type 'custom' to enter your own bug report\n")
    
    while True:
        choice = input("Your choice: ").strip().lower()
        
        if choice in ['quit', 'exit', 'q']:
            print("\n👋 Thanks for using the classifier!")
            break
        
        if choice.isdigit() and 1 <= int(choice) <= len(examples):
            idx = int(choice) - 1
            title = examples[idx]['title']
            description = examples[idx]['description']
            print(f"\n📋 Selected: {title}\n")
        elif choice == 'custom':
            print("\n--- Enter Bug Details ---")
            title = input("Bug Title: ").strip()
            if not title:
                print("❌ Title cannot be empty!\n")
                continue
            description = input("Bug Description: ").strip()
            if not description:
                print("❌ Description cannot be empty!\n")
                continue
            print()
        else:
            print("❌ Invalid choice. Please try again.\n")
            continue
        
        # Classify the bug
        result = classify_bug(model, tokenizer, title, description)
        
        print("="*60)
        print("📊 CLASSIFICATION RESULT")
        print("="*60)
        print(result)
        print("="*60 + "\n")


def single_prediction(model, tokenizer, title, description):
    """Make a single prediction (non-interactive)"""
    result = classify_bug(model, tokenizer, title, description)
    
    print("="*60)
    print("📊 CLASSIFICATION RESULT")
    print("="*60)
    print(f"\n{result}\n")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--base_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--title", type=str, help="Bug title (for single prediction)")
    parser.add_argument("--description", type=str, help="Bug description (for single prediction)")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model_path, args.base_model)
    
    # Run demo
    if args.title and args.description:
        # Single prediction mode
        single_prediction(model, tokenizer, args.title, args.description)
    else:
        # Interactive mode
        interactive_demo(model, tokenizer)
