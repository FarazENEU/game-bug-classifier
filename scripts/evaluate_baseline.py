"""
Evaluate Zero-Shot Baseline (Mistral base model with no fine-tuning)
Critical for rubric: shows improvement from fine-tuning
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import argparse
from datetime import datetime


def load_base_model(model_name):
    """Load base Mistral model without any fine-tuning"""
    print(f"\n🔍 Loading base model: {model_name}")
    print("⚠️  NOTE: This is the PRE-FINE-TUNED model for baseline comparison\n")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Use 4-bit quantization to fit in memory
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
    model.eval()
    
    print("✅ Base model loaded (no fine-tuning applied)\n")
    return model, tokenizer


def format_prompt(title, description):
    """Same prompt format as fine-tuned model"""
    instruction = "Classify this bug report and provide a structured analysis."
    input_text = f"Title: {title}\n\nDescription: {description}"
    
    prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Output:
"""
    return prompt


def parse_output(output):
    """Parse model output - handles both structured and unstructured responses"""
    result = {
        'severity': None,
        'component': None,
        'reproducibility': None,
        'raw': output
    }
    
    # Try to extract structured fields
    for line in output.split('\n'):
        line = line.strip()
        if 'severity:' in line.lower():
            parts = line.split(':', 1)
            if len(parts) == 2:
                result['severity'] = parts[1].strip()
        elif 'component:' in line.lower():
            parts = line.split(':', 1)
            if len(parts) == 2:
                result['component'] = parts[1].strip()
        elif 'reproducibility:' in line.lower():
            parts = line.split(':', 1)
            if len(parts) == 2:
                result['reproducibility'] = parts[1].strip()
    
    return result


def normalize_label(label, field_type):
    """Normalize labels to match expected format"""
    if label is None:
        return None
    
    label = label.lower().strip()
    
    if field_type == 'severity':
        # Map variations to standard labels
        if 'critical' in label or 'crit' in label:
            return 'critical'
        elif 'high' in label:
            return 'high'
        elif 'medium' in label or 'med' in label or 'moderate' in label:
            return 'medium'
        elif 'low' in label or 'minor' in label or 'trivial' in label:
            return 'low'
    
    elif field_type == 'reproducibility':
        if 'always' in label or 'every' in label or '100%' in label:
            return 'always'
        elif 'sometimes' in label or 'intermittent' in label or 'occasional' in label:
            return 'sometimes'
        elif 'rare' in label or 'rarely' in label or 'hard' in label:
            return 'rare'
    
    elif field_type == 'component':
        # Return first word for component (usually correct)
        words = label.split()
        if words:
            comp = words[0].lower()
            # Common components
            valid = ['ui', 'gameplay', 'audio', 'graphics', 'network', 'save', 
                    'physics', 'animation', 'ai', 'multiplayer', 'performance']
            if any(v in comp for v in valid):
                return comp
    
    return None


def evaluate_baseline(model, tokenizer, test_path, num_samples=100):
    """Evaluate zero-shot baseline"""
    print(f"📊 Evaluating zero-shot baseline on {num_samples} samples...")
    print("="*70)
    
    # Load test data
    with open(test_path, 'r') as f:
        test_data = [json.loads(line) for line in f]
    
    if num_samples > 0:
        test_data = test_data[:num_samples]
    
    results = {
        'severity': {'correct': 0, 'total': 0},
        'component': {'correct': 0, 'total': 0},
        'reproducibility': {'correct': 0, 'total': 0},
        'predictions': []
    }
    
    for i, example in enumerate(tqdm(test_data, desc="Evaluating")):
        # Parse expected output
        expected_output = example['output'].strip()
        expected = {}
        for line in expected_output.split('\n'):
            if line.startswith('Severity:'):
                expected['severity'] = line.split(':', 1)[1].strip().lower()
            elif line.startswith('Component:'):
                expected['component'] = line.split(':', 1)[1].strip().lower()
            elif line.startswith('Reproducibility:'):
                expected['reproducibility'] = line.split(':', 1)[1].strip().lower()
        
        # Get model prediction
        prompt = format_prompt(example['title'], example['description'])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                temperature=1.0,
            )
        
        generated = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        parsed = parse_output(generated)
        
        # Normalize and compare
        pred_result = {
            'id': i,
            'title': example['title'],
            'expected': expected,
            'predicted_raw': parsed,
            'predicted_normalized': {},
            'correct': {}
        }
        
        for field in ['severity', 'component', 'reproducibility']:
            if field in expected:
                results[field]['total'] += 1
                
                # Normalize prediction
                pred_normalized = normalize_label(parsed[field], field)
                pred_result['predicted_normalized'][field] = pred_normalized
                
                # Check correctness
                is_correct = (pred_normalized == expected[field] if pred_normalized else False)
                pred_result['correct'][field] = is_correct
                
                if is_correct:
                    results[field]['correct'] += 1
        
        results['predictions'].append(pred_result)
    
    # Calculate accuracies
    print("\n" + "="*70)
    print("📈 ZERO-SHOT BASELINE RESULTS")
    print("="*70)
    
    accuracies = {}
    for field in ['severity', 'component', 'reproducibility']:
        if results[field]['total'] > 0:
            acc = (results[field]['correct'] / results[field]['total']) * 100
            accuracies[field] = acc
            print(f"{field.capitalize():15} {acc:6.2f}% ({results[field]['correct']}/{results[field]['total']})")
    
    # Overall accuracy
    total_correct = sum(r['correct'] for r in results.values() if isinstance(r, dict) and 'correct' in r)
    total_predictions = sum(r['total'] for r in results.values() if isinstance(r, dict) and 'total' in r)
    overall_acc = (total_correct / total_predictions * 100) if total_predictions > 0 else 0
    accuracies['overall'] = overall_acc
    
    print(f"{'Overall':15} {overall_acc:6.2f}% ({total_correct}/{total_predictions})")
    print("="*70)
    
    results['accuracies'] = accuracies
    results['metadata'] = {
        'num_samples': len(test_data),
        'timestamp': datetime.now().isoformat(),
        'model': 'Zero-shot baseline (no fine-tuning)',
        'note': 'This is the PRE-FINE-TUNED model for comparison'
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate zero-shot baseline")
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Base model name (no fine-tuning)"
    )
    parser.add_argument(
        "--test_path",
        type=str,
        required=True,
        help="Path to test.jsonl"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate (0 for all)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="baseline_zero_shot_results.json",
        help="Output JSON file"
    )
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_base_model(args.model_name)
    
    # Evaluate
    results = evaluate_baseline(model, tokenizer, args.test_path, args.num_samples)
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {args.output_file}")
    print("\n✅ Zero-shot baseline evaluation complete!")
    print("📊 Compare these results against your fine-tuned models to show improvement\n")


if __name__ == "__main__":
    main()
