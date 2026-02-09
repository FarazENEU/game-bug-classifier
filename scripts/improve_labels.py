"""
Label Quality Improvement Script
Uses Mistral-7B-Instruct base model to generate better labels
Run this on Kaggle (free GPU) to improve label quality before retraining
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

def load_dataset(path):
    """Load JSONL dataset"""
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def improve_labels_with_mistral(bug_report, model, tokenizer):
    """
    Use Mistral-7B-Instruct base model to generate better labels
    This is the instruction-tuned model (smarter than keywords) but NOT our fine-tuned version
    This creates a "bootstrapping" effect - using strong base model to improve training labels
    """
    
    instruction = """Classify this game engine bug report. Provide labels in exactly this format:
Severity: [critical/major/moderate/minor]
Component: [rendering/physics/ui/networking/audio/gameplay/other]
Reproducibility: [always/often/sometimes/rare]

Guidelines:
- critical: crashes, data loss, security, game-breaking
- major: broken features, significant gameplay impact
- moderate: noticeable issues with workarounds
- minor: cosmetic, polish, minor annoyances

- always: 100% reproduction
- often: >70% reproduction  
- sometimes: 30-70% reproduction
- rare: <30% reproduction or unclear steps

Base classification on context and impact, not just keywords."""
    
    prompt = f"""### Instruction:
{instruction}

### Input:
{bug_report}

### Output:
"""
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.3,  # Low temperature for consistency
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the generated part
        generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return generated.strip()
    
    except Exception as e:
        print(f"Error generating labels: {e}")
        return None


def main():
    # Configuration
    MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
    TRAIN_PATH = "/kaggle/input/data-splits/train.jsonl"
    OUTPUT_PATH = "/kaggle/working/train_improved.jsonl"
    
    print("="*70)
    print("🔬 LABEL QUALITY IMPROVEMENT (FREE)")
    print("="*70)
    print(f"\nThis will re-label {TRAIN_PATH} using Mistral-7B-Instruct base model")
    print(f"Output: {OUTPUT_PATH}")
    print("\nMethod: Bootstrapping with instruction-tuned base model")
    print("Cost: FREE (Kaggle GPU)")
    print("Estimated time: ~30-40 minutes")
    
    # Load training data
    print(f"\n📂 Loading training data from {TRAIN_PATH}...")
    train_data = load_dataset(TRAIN_PATH)
    print(f"✅ Loaded {len(train_data)} training examples")
    
    # Load Mistral base model
    print(f"\n🔍 Loading Mistral-7B-Instruct base model...")
    print("(This will be cached after first run on Kaggle)")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Use 4-bit quantization to save memory
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("✅ Model loaded")
    
    # Improve labels
    print(f"\n🚀 Generating improved labels with base Mistral model...")
    improved_data = []
    failed = 0
    
    for example in tqdm(train_data, desc="Improving labels"):
        # Get Mistral-generated labels
        improved_output = improve_labels_with_mistral(example['input'], model, tokenizer)
        
        if improved_output:
            # Create new example with improved labels
            improved_example = {
                'instruction': example['instruction'],
                'input': example['input'],
                'output': improved_output,
                'metadata': {
                    'original_output': example['output'],
                    'improvement_method': 'mistral-base-model',
                    'original_repo': example.get('metadata', {}).get('repo', 'unknown')
                }
            }
            improved_data.append(improved_example)
        else:
            failed += 1
            # Keep original if generation fails
            improved_data.append(example)
    
    # Save improved dataset
    print(f"\n💾 Saving improved dataset to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        for example in improved_data:
            f.write(json.dumps(example) + '\n')
    
    print(f"✅ Saved {len(improved_data)} examples")
    print(f"⚠️  Failed to improve {failed} examples (kept original labels)")
    
    # Show comparison
    print("\n" + "="*70)
    print("📊 LABEL COMPARISON (First 3 examples)")
    print("="*70)
    
    for i in range(min(3, len(improved_data))):
        print(f"\n--- Example {i+1} ---")
        print(f"Input: {improved_data[i]['input'][:150]}...")
        print(f"\nOriginal Labels:\n{improved_data[i]['metadata']['original_output']}")
        print(f"\nImproved Labels (GPT-4):\n{improved_data[i]['output']}")
        print("-" * 70)
    
    print("\n" + "="*70)
    print("✨ Label improvement complete!")
    print("="*70)
    print(f"\nNext steps:")
    print(f"1. Review {OUTPUT_PATH} to verify quality")
    print(f"2. Retrain with improved labels:")
    print(f"   python train.py (will use train_improved.jsonl)")
    print(f"3. Compare old vs new model performance")
    print(f"\nBootstrapping approach:")
    print(f"   - Used instruction-tuned base model (smarter than keywords)")
    print(f"   - Generated contextual labels instead of keyword matching")
    print(f"   - Should improve fine-tuned model quality")


if __name__ == "__main__":
    main()
