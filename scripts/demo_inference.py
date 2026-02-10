"""
Quick inference demo for video walkthrough
Shows the fine-tuned model classifying a sample bug report
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import json

# Example bug report for demo
SAMPLE_BUG = """
Application crashes immediately after clicking the "Start Game" button.
Error message: "Segmentation fault in RenderEngine.cpp line 142".
This happens 100% of the time on both Windows and Linux.
Affects all users who updated to the latest version.
"""

def main():
    print("=" * 70)
    print("Bug Classification Demo - V1 Model (Mistral-7B + QLoRA)")
    print("Results: 64.34% avg accuracy (41.86% severity, 62.79% component, 88.37% reproducibility)")
    print("=" * 70)
    
    print("\n📝 Sample Bug Report:")
    print("-" * 70)
    print(SAMPLE_BUG.strip())
    print("-" * 70)
    
    # Load model
    print("\n🔄 Loading fine-tuned model...")
    base_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    adapter_path = "final_model"  # Update if different
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
        print("✅ Model loaded successfully!")
        
        # Create classification prompt
        prompt = f"""You are a bug triaging assistant. Classify this bug report.

Bug Report:
{SAMPLE_BUG.strip()}

Classification (provide ONLY the labels, one per line):
Severity:
Component:
Reproducibility:"""

        # Generate prediction
        print("\n🤖 Model Prediction:")
        print("-" * 70)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the classification part
        if "Classification" in response:
            classification = response.split("Classification")[-1].strip()
            print(classification)
        else:
            print(response)
        
        print("-" * 70)
        
        print("\n💡 Expected Classification:")
        print("   Severity: critical (crashes, segmentation fault)")
        print("   Component: graphics (RenderEngine error)")
        print("   Reproducibility: always (100% reproduction rate)")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\n📌 Note: Make sure adapter path is correct")
        print(f"   Looking for: {adapter_path}")
        print("\n   For demo, you can show the concept without actual inference")
    
    print("\n" + "=" * 70)
    print("Demo complete! This shows the model's multi-task classification ability.")
    print("=" * 70)

if __name__ == "__main__":
    main()
