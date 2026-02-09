"""
Compare results from all hyperparameter experiments
Loads all evaluation JSON files and displays comparison table
"""

import json
import os
from pathlib import Path

def load_results(filepath):
    """Load evaluation results from JSON"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            return data.get('accuracies', {})
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"⚠️ Error loading {filepath}: {e}")
        return None

def main():
    print("\n" + "="*80)
    print("📊 HYPERPARAMETER OPTIMIZATION RESULTS COMPARISON")
    print("="*80)
    
    # Define all configurations
    configs = [
        {
            'name': 'Zero-shot (base)',
            'file': 'baseline_zero_shot_results.json',
            'r': 0,
            'alpha': 0,
            'params': '0'
        },
        {
            'name': 'V1 (keyword)',
            'file': None,  # Manual data
            'r': 8,
            'alpha': 32,
            'params': '16M',
            'manual': {
                'severity': 41.86,
                'component': 62.79,
                'reproducibility': 88.37,
                'overall': 64.34
            }
        },
        {
            'name': 'V2-r4',
            'file': 'evaluation_results_v2_r4.json',
            'r': 4,
            'alpha': 16,
            'params': '8M'
        },
        {
            'name': 'V2-r8 (baseline)',
            'file': 'evaluation_results_v2_r8.json',
            'r': 8,
            'alpha': 32,
            'params': '16M'
        },
        {
            'name': 'V2-r16',
            'file': 'evaluation_results_v2_r16.json',
            'r': 16,
            'alpha': 64,
            'params': '32M'
        }
    ]
    
    # Load all results
    results = []
    for config in configs:
        if config.get('manual'):
            results.append({
                'name': config['name'],
                'r': config['r'],
                'alpha': config['alpha'],
                'params': config['params'],
                'metrics': config['manual'],
                'status': 'MANUAL'
            })
        elif config['file']:
            metrics = load_results(config['file'])
            if metrics:
                results.append({
                    'name': config['name'],
                    'r': config['r'],
                    'alpha': config['alpha'],
                    'params': config['params'],
                    'metrics': metrics,
                    'status': 'LOADED'
                })
            else:
                results.append({
                    'name': config['name'],
                    'r': config['r'],
                    'alpha': config['alpha'],
                    'params': config['params'],
                    'metrics': None,
                    'status': 'MISSING'
                })
    
    # Display table
    print("\n📋 RESULTS TABLE:\n")
    
    # Header
    print(f"{'Config':<20} {'r':>3} {'α':>4} {'Params':>7} {'Overall':>9} {'Severity':>9} {'Component':>10} {'Repro':>9}")
    print("-" * 80)
    
    # Rows
    for r in results:
        name = r['name']
        lora_r = r['r'] if r['r'] > 0 else 'N/A'
        alpha = r['alpha'] if r['alpha'] > 0 else 'N/A'
        params = r['params']
        
        if r['metrics']:
            overall = f"{r['metrics'].get('overall', 0):.2f}%"
            severity = f"{r['metrics'].get('severity', 0):.2f}%"
            component = f"{r['metrics'].get('component', 0):.2f}%"
            repro = f"{r['metrics'].get('reproducibility', 0):.2f}%"
        else:
            overall = severity = component = repro = "MISSING"
        
        print(f"{name:<20} {str(lora_r):>3} {str(alpha):>4} {params:>7} {overall:>9} {severity:>9} {component:>10} {repro:>9}")
    
    print("\n" + "="*80)
    
    # Analysis
    print("\n🔍 QUICK ANALYSIS:\n")
    
    # Find best overall
    valid_results = [r for r in results if r['metrics'] and 'V2-' in r['name']]
    if valid_results:
        best = max(valid_results, key=lambda x: x['metrics'].get('overall', 0))
        print(f"✅ Best overall accuracy: {best['name']} ({best['metrics']['overall']:.2f}%)")
        
        # Find V1 baseline
        v1 = next((r for r in results if 'V1' in r['name']), None)
        if v1 and v1['metrics']:
            improvement = best['metrics']['overall'] - v1['metrics']['overall']
            print(f"📈 Improvement over V1: +{improvement:.2f} percentage points")
            
            severity_improvement = best['metrics']['severity'] - v1['metrics']['severity']
            print(f"📈 Severity improvement: {v1['metrics']['severity']:.2f}% → {best['metrics']['severity']:.2f}% (+{severity_improvement:.2f}pp)")
    
    # Check for overfitting signs
    print("\n⚠️ OVERFITTING ANALYSIS:")
    v2_models = [r for r in valid_results if r['name'] != 'V2-r8 (baseline)']
    baseline = next((r for r in results if 'V2-r8' in r['name']), None)
    
    if baseline and baseline['metrics'] and v2_models:
        for model in v2_models:
            if model['metrics']['overall'] < baseline['metrics']['overall']:
                print(f"   ⚠️ {model['name']}: Lower accuracy than baseline (possible {'underfit' if model['r'] < baseline['r'] else 'overfit'})")
            else:
                print(f"   ✅ {model['name']}: {'Higher' if model['metrics']['overall'] > baseline['metrics']['overall'] else 'Similar'} accuracy")
    
    # Missing files warning
    missing = [r for r in results if r['status'] == 'MISSING']
    if missing:
        print("\n⚠️ MISSING FILES:")
        for m in missing:
            expected_file = next((c['file'] for c in configs if c['name'] == m['name']), 'unknown')
            print(f"   ❌ {m['name']}: {expected_file}")
    
    # Markdown table for report
    print("\n" + "="*80)
    print("📝 MARKDOWN TABLE (copy to HYPERPARAMETER_OPTIMIZATION.md):\n")
    print("| Config              | r   | α   | Params | Overall | Severity | Component | Repro  |")
    print("|---------------------|-----|-----|--------|---------|----------|-----------|--------|")
    for r in results:
        name = r['name']
        lora_r = r['r'] if r['r'] > 0 else '-'
        alpha = r['alpha'] if r['alpha'] > 0 else '-'
        params = r['params']
        
        if r['metrics']:
            overall = f"{r['metrics'].get('overall', 0):.2f}%"
            severity = f"{r['metrics'].get('severity', 0):.2f}%"
            component = f"{r['metrics'].get('component', 0):.2f}%"
            repro = f"{r['metrics'].get('reproducibility', 0):.2f}%"
        else:
            overall = severity = component = repro = "-"
        
        print(f"| {name:<19} | {str(lora_r):<3} | {str(alpha):<3} | {params:<6} | {overall:<7} | {severity:<8} | {component:<9} | {repro:<6} |")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
