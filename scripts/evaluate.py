"""
Evaluation Script for Bug Report Classifier
Tests the fine-tuned model on the test set
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import argparse
from collections import defaultdict


def load_test_data(test_path):
    """Load test data from JSONL file"""
    test_data = []
    with open(test_path, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))
    return test_data


def generate_prediction(model, tokenizer, instruction, input_text, max_length=512):
    """Generate prediction for a single example"""
    prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Output:
"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the generated part (skip the input prompt)
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return generated_text.strip()


def parse_output(output_text):
    """Parse the model output to extract structured fields"""
    fields = {}
    
    # Try to extract severity
    if "Severity:" in output_text:
        start = output_text.index("Severity:") + len("Severity:")
        end = output_text.find("\n", start)
        if end == -1:
            end = len(output_text)
        fields['severity'] = output_text[start:end].strip().lower()
    
    # Try to extract component
    if "Component:" in output_text:
        start = output_text.index("Component:") + len("Component:")
        end = output_text.find("\n", start)
        if end == -1:
            end = len(output_text)
        fields['component'] = output_text[start:end].strip().lower()
    
    # Try to extract reproducibility
    if "Reproducibility:" in output_text:
        start = output_text.index("Reproducibility:") + len("Reproducibility:")
        end = output_text.find("\n", start)
        if end == -1:
            end = len(output_text)
        fields['reproducibility'] = output_text[start:end].strip().lower()
    
    return fields


def calculate_accuracy(predictions, ground_truths, field):
    """Calculate accuracy for a specific field"""
    correct = 0
    total = 0
    
    for pred, gt in zip(predictions, ground_truths):
        pred_fields = parse_output(pred)
        gt_fields = parse_output(gt)
        
        if field in pred_fields and field in gt_fields:
            if pred_fields[field] == gt_fields[field]:
                correct += 1
            total += 1
    
    return (correct / total * 100) if total > 0 else 0.0


def evaluate(model_path, base_model_name, test_path, num_samples=None):
    """Evaluate the fine-tuned model"""
    print(f"🔍 Loading model from {model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Load fine-tuned adapter
    print("Loading fine-tuned adapter...")
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    # Load test data
    print(f"📂 Loading test data from {test_path}...")
    test_data = load_test_data(test_path)
    
    if num_samples:
        test_data = test_data[:num_samples]
    
    print(f"Testing on {len(test_data)} examples...")
    
    # Generate predictions
    predictions = []
    ground_truths = []
    
    for example in tqdm(test_data, desc="Generating predictions"):
        instruction = example['instruction']
        input_text = example['input']
        expected_output = example['output']
        
        prediction = generate_prediction(model, tokenizer, instruction, input_text)
        
        predictions.append(prediction)
        ground_truths.append(expected_output)
    
    # Calculate metrics
    print("\n" + "="*50)
    print("📊 EVALUATION RESULTS")
    print("="*50)
    
    severity_acc = calculate_accuracy(predictions, ground_truths, 'severity')
    component_acc = calculate_accuracy(predictions, ground_truths, 'component')
    reproducibility_acc = calculate_accuracy(predictions, ground_truths, 'reproducibility')
    
    print(f"\nSeverity Accuracy: {severity_acc:.2f}%")
    print(f"Component Accuracy: {component_acc:.2f}%")
    print(f"Reproducibility Accuracy: {reproducibility_acc:.2f}%")
    print(f"Overall Average: {(severity_acc + component_acc + reproducibility_acc) / 3:.2f}%")
    
    # Show some examples
    print("\n" + "="*50)
    print("📝 SAMPLE PREDICTIONS (First 3)")
    print("="*50)
    
    for i in range(min(3, len(test_data))):
        print(f"\n--- Example {i+1} ---")
        print(f"Input: {test_data[i]['input'][:200]}...")
        print(f"\nExpected:\n{ground_truths[i]}")
        print(f"\nPredicted:\n{predictions[i]}")
        print("-" * 50)
    
    # Save full results
    results_path = os.path.join(os.path.dirname(model_path), "evaluation_results.json")
    results = {
        "metrics": {
            "severity_accuracy": severity_acc,
            "component_accuracy": component_acc,
            "reproducibility_accuracy": reproducibility_acc,
            "overall_average": (severity_acc + component_acc + reproducibility_acc) / 3
        },
        "num_samples": len(test_data),
        "predictions": [
            {
                "input": test_data[i]['input'][:100],
                "expected": ground_truths[i],
                "predicted": predictions[i]
            }
            for i in range(len(test_data))
        ]
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Full results saved to: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--base_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--test_path", type=str, default="/kaggle/input/data-split/test.jsonl")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to test (default: all)")
    
    args = parser.parse_args()
    
    evaluate(
        model_path=args.model_path,
        base_model_name=args.base_model,
        test_path=args.test_path,
        num_samples=args.num_samples
    )
