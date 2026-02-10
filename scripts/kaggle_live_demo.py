"""
Live Inference Demo for Kaggle Video Recording
Shows the fine-tuned model classifying a sample bug report in real-time
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
import os

# Sample bug for live demo - graphics crash
DEMO_BUG_TITLE = "Game crashes when loading high-res textures"
DEMO_BUG_DESCRIPTION = """
The game crashes to desktop whenever I enter the forest area which has 
high-resolution tree textures. I get a "GPU memory allocation failed" error 
in the console. This happens every single time I try to enter that zone.
Works fine in all other areas with lower quality textures.
"""

def main():
    print("\n" + "="*70)
    print("🎬 LIVE INFERENCE DEMO - Bug Classification with QLoRA Fine-Tuned Mistral-7B")
    print("="*70)
    print(f"Current directory: {os.getcwd()}")
    
    # Show the sample bug
    print("\n📝 Sample Bug Report:")
    print("-"*70)
    print(f"Title: {DEMO_BUG_TITLE}")
    print(f"\nDescription: {DEMO_BUG_DESCRIPTION.strip()}")
    print("-"*70)
    
    # Load model (this part can be edited out of video)
    print("\n🔄 Loading fine-tuned model...")
    print("   Base: mistralai/Mistral-7B-Instruct-v0.2")
    print("   Method: QLoRA (4-bit quantization)")
    
    # Check multiple possible locations for the model
    possible_paths = [
        "/kaggle/working/game-bug-classifier/final_model/outputs/final_model",  # Full path in cloned repo
        "/kaggle/working/final_model/outputs/final_model",  # If uploaded separately
        "/kaggle/input/llm-bug-classifier-v1/final_model/outputs/final_model",  # If added as dataset
        os.path.join(os.getcwd(), "final_model/outputs/final_model"),  # Relative to current directory
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            # Check if adapter files exist
            adapter_config = os.path.join(path, "adapter_config.json")
            adapter_model = os.path.join(path, "adapter_model.safetensors")
            if os.path.exists(adapter_config) and os.path.exists(adapter_model):
                model_path = path
                print(f"   Found model at: {path}")
                break
    
    if not model_path:
        print("\n❌ Model not found! Tried:")
        for path in possible_paths:
            exists = "✓" if os.path.exists(path) else "✗"
            print(f"   {exists} {path}")
        print("\n💡 Make sure you're running from the repository directory:")
        print("   cd /kaggle/working/game-bug-classifier")
        print("   python scripts/kaggle_live_demo.py")
        return
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model with 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load fine-tuned adapter
        model = PeftModel.from_pretrained(
            base_model, 
            model_path
        )
        model.eval()
        
        print("✅ Model loaded successfully!\n")
        
        # Create the input prompt in the format model was trained on
        input_text = f"Title: {DEMO_BUG_TITLE}\n\nDescription: {DEMO_BUG_DESCRIPTION.strip()}"
        
        prompt = f"""### Instruction:
Classify this bug report into severity, component, and reproducibility.

### Input:
{input_text}

### Output:
"""
        
        # Tokenize
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate classification
        print("🤖 Running inference...")
        print("   (Generating classification...)\n")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,  # Low temperature for consistent output
                do_sample=False,   # Greedy decoding for deterministic results
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode the output
        generated_text = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        # Display results
        print("="*70)
        print("📊 MODEL PREDICTION")
        print("="*70)
        print(generated_text.strip())
        print("="*70)
        
        # Show expected classification for comparison
        print("\n💡 Expected Classification (based on training patterns):")
        print("   Severity: high (crashes, affects gameplay)")
        print("   Component: graphics (GPU memory, textures)")
        print("   Reproducibility: always (100% reproduction rate)")
        
        print("\n" + "="*70)
        print("✅ Demo Complete!")
        print(f"   Model: V1-r8 (64.34% average accuracy)")
        print(f"   Training: 1,399 examples from 4 game repos")
        print(f"   Method: QLoRA with rank=8, 16M trainable params")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        print("   This script requires GPU and the transformers/peft libraries.")
        print("   Make sure you're running on a Kaggle GPU notebook.")

if __name__ == "__main__":
    main()
