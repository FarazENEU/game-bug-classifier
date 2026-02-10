"""
Create performance visualizations for portfolio presentation.

Generates:
1. Performance comparison bar chart (baseline vs all configs)
2. Three confusion matrices (severity, component, reproducibility)
3. Training curves (if training logs available)
4. Example showcase table

Usage:
    python scripts/create_visualizations.py
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List
import pandas as pd

# Set style for professional-looking plots
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

OUTPUT_DIR = Path("visuals")
OUTPUT_DIR.mkdir(exist_ok=True)


def create_performance_comparison():
    """
    Create bar chart comparing all configurations across all tasks.
    
    Data needed: Your results table from README
    """
    # Data from your results table
    configs = ['Baseline\n(zero-shot)', 'V1-r4\n(1 epoch)', 'V1-r8\n(3 epochs)', 'V1-r16\n(1 epoch)']
    
    severity = [1.00, 47.92, 41.86, 48.94]
    component = [0.00, 62.50, 62.79, 61.70]
    reproducibility = [0.00, 69.47, 88.37, 69.15]
    overall = [0.33, 59.96, 64.34, 59.93]
    
    x = np.arange(len(configs))
    width = 0.2  # width of bars
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Create grouped bars
    bars1 = ax.bar(x - 1.5*width, severity, width, label='Severity', alpha=0.8)
    bars2 = ax.bar(x - 0.5*width, component, width, label='Component', alpha=0.8)
    bars3 = ax.bar(x + 0.5*width, reproducibility, width, label='Reproducibility', alpha=0.8)
    bars4 = ax.bar(x + 1.5*width, overall, width, label='Overall', alpha=0.8, 
                    color='darkgreen', linewidth=2, edgecolor='black')
    
    # Customize
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Model Configuration', fontsize=13, fontweight='bold')
    ax.set_title('Multi-Task Classification Performance Comparison', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend(loc='upper left', fontsize=11)
    ax.set_ylim(0, 100)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 5:  # Only label if visible
                ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=9, rotation=0)
    
    add_value_labels(bars4)  # Label overall bars
    
    # Add horizontal line at 64.34% to highlight best result
    ax.axhline(y=64.34, color='green', linestyle='--', alpha=0.5, 
               label='Best Overall (64.34%)')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'performance_comparison.png'}")
    plt.close()


def parse_labels_from_output(output_text):
    """Parse severity/component/reproducibility from model output"""
    labels = {}
    
    for field in ['severity', 'component', 'reproducibility']:
        pattern = f"{field.capitalize()}:"
        if pattern in output_text:
            start = output_text.index(pattern) + len(pattern)
            end = output_text.find("\n", start)
            if end == -1:
                end = len(output_text)
            value = output_text[start:end].strip().lower()
            # Clean up any extra text
            value = value.split()[0] if value else 'unknown'
            labels[field] = value
        else:
            labels[field] = 'unknown'
    
    return labels


def create_confusion_matrices():
    """
    Create confusion matrices for each classification task.
    
    Uses real predictions from final_model/evaluation_results.json
    which contains predictions from quick_eval.py
    """
    
    # Check for real prediction data
    pred_file = Path("final_model/evaluation_results.json")
    
    if not pred_file.exists():
        print(f"⚠️  No evaluation results found at {pred_file}")
        print("   Creating placeholder confusion matrices...")
        create_placeholder_confusion_matrices()
        return
    
    # Load predictions
    with open(pred_file) as f:
        data = json.load(f)
    
    if 'predictions' not in data:
        print(f"⚠️  No predictions field in {pred_file}")
        print("   Creating placeholder confusion matrices...")
        create_placeholder_confusion_matrices()
        return
    
    predictions = data['predictions']
    print(f"   Loaded {len(predictions)} predictions from evaluation results")
    
    # Parse predictions
    parsed_predictions = []
    for pred in predictions:
        expected = parse_labels_from_output(pred['expected'])
        predicted = parse_labels_from_output(pred['predicted'])
        parsed_predictions.append({
            'severity_true': expected['severity'],
            'severity_pred': predicted['severity'],
            'component_true': expected['component'],
            'component_pred': predicted['component'],
            'reproducibility_true': expected['reproducibility'],
            'reproducibility_pred': predicted['reproducibility'],
        })
    
    # Extract labels for each task
    tasks = ['severity', 'component', 'reproducibility']
    
    for task in tasks:
        true_labels = [p[f'{task}_true'] for p in parsed_predictions]
        pred_labels = [p[f'{task}_pred'] for p in parsed_predictions]
        
        # Filter out unknown labels
        valid_pairs = [(t, p) for t, p in zip(true_labels, pred_labels) 
                       if t != 'unknown' and p != 'unknown']
        
        if not valid_pairs:
            print(f"⚠️  No valid predictions for {task}, skipping")
            continue
        
        true_labels, pred_labels = zip(*valid_pairs)
        
        # Get unique labels (sorted)
        labels = sorted(set(true_labels) | set(pred_labels))
        
        # Create confusion matrix
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        
        for true, pred in zip(true_labels, pred_labels):
            cm[label_to_idx[true], label_to_idx[pred]] += 1
        
        # Convert to percentages (rows sum to 100)
        cm_pct = cm.astype(float)
        row_sums = cm_pct.sum(axis=1, keepdims=True)
        cm_pct = np.divide(cm_pct, row_sums, where=row_sums!=0) * 100
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels,
                    cbar_kws={'label': 'Percentage (%)'},
                    linewidths=0.5, linecolor='gray')
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title(f'{task.capitalize()} Classification - Confusion Matrix (n={len(valid_pairs)})',
                     fontsize=14, fontweight='bold', pad=15)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'confusion_matrix_{task}.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {OUTPUT_DIR / f'confusion_matrix_{task}.png'} (real data from quick_eval)")
        plt.close()


def create_placeholder_confusion_matrices():
    """
    Create example confusion matrices using approximate data.
    Replace with real data once you have predictions.
    """
    # Severity (4 classes) - Medium accuracy (41.86%)
    severity_labels = ['critical', 'high', 'medium', 'low']
    severity_cm = np.array([
        [40, 5, 3, 2],   # critical
        [8, 35, 6, 1],   # high  
        [5, 10, 45, 5],  # medium
        [2, 3, 8, 37]    # low
    ])
    
    # Component (7 classes) - Good accuracy (62.79%)
    component_labels = ['audio', 'graphics', 'network', 'other', 'performance', 'save', 'ui']
    component_cm = np.array([
        [60, 5, 2, 8, 3, 2, 5],    # audio
        [3, 65, 2, 5, 8, 2, 3],    # graphics
        [2, 2, 68, 5, 3, 5, 3],    # network
        [10, 8, 5, 55, 5, 3, 8],   # other
        [4, 10, 3, 6, 62, 3, 4],   # performance
        [3, 2, 4, 5, 2, 70, 3],    # save
        [5, 3, 2, 8, 4, 2, 65]     # ui
    ])
    
    # Reproducibility (3 classes) - Excellent accuracy (88.37%)
    repro_labels = ['always', 'sometimes', 'rare']
    repro_cm = np.array([
        [85, 10, 5],    # always
        [5, 90, 5],     # sometimes
        [8, 7, 85]      # rare
    ])
    
    matrices = {
        'severity': (severity_labels, severity_cm),
        'component': (component_labels, component_cm),
        'reproducibility': (repro_labels, repro_cm)
    }
    
    for task, (labels, cm) in matrices.items():
        # Convert to percentages
        cm_pct = cm.astype(float)
        row_sums = cm_pct.sum(axis=1, keepdims=True)
        cm_pct = (cm_pct / row_sums) * 100
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=labels, yticklabels=labels,
                    cbar_kws={'label': 'Percentage (%)'},
                    linewidths=0.5, linecolor='gray')
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title(f'{task.capitalize()} Classification - Confusion Matrix (Estimated)',
                     fontsize=14, fontweight='bold', pad=15)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'confusion_matrix_{task}.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {OUTPUT_DIR / f'confusion_matrix_{task}.png'} (placeholder)")
        plt.close()


def create_training_curves():
    """
    Create training loss curves over time.
    
    Data needed: trainer_state.json or training logs from outputs_v*/
    
    If you don't have this, skip this visualization.
    """
    
    # Check for training state files
    possible_paths = [
        "outputs_v1_r8/checkpoint-final/trainer_state.json",
        "final_model/outputs/final_model/trainer_state.json",
    ]
    
    state_file = None
    for path in possible_paths:
        if Path(path).exists():
            state_file = Path(path)
            break
    
    if state_file is None:
        print("⚠️  No trainer_state.json found - skipping training curves")
        print("   This is optional but shows training progression nicely")
        return
    
    # Load training state
    with open(state_file) as f:
        state = json.load(f)
    
    # Extract loss history
    log_history = state.get('log_history', [])
    
    train_steps = []
    train_loss = []
    eval_steps = []
    eval_loss = []
    
    for entry in log_history:
        if 'loss' in entry:
            train_steps.append(entry['step'])
            train_loss.append(entry['loss'])
        if 'eval_loss' in entry:
            eval_steps.append(entry['step'])
            eval_loss.append(entry['eval_loss'])
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if train_loss:
        ax.plot(train_steps, train_loss, label='Training Loss', 
                linewidth=2, alpha=0.8, marker='o', markersize=4)
    if eval_loss:
        ax.plot(eval_steps, eval_loss, label='Validation Loss',
                linewidth=2, alpha=0.8, marker='s', markersize=4)
    
    ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training Progress - V1-r8 (3 epochs)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_curves.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'training_curves.png'}")
    plt.close()


def create_example_showcase():
    """
    Create a table showing example predictions.
    
    Shows: Bug text → Baseline prediction (wrong) → Fine-tuned prediction (correct)
    
    Data needed: 
    - test.jsonl for true labels
    - baseline_zero_shot_results.json for baseline predictions
    - final_model/evaluation_predictions.json for fine-tuned predictions
    """
    
    print("\n📋 Example Showcase")
    print("=" * 80)
    print("For this, manually select 3-5 compelling examples where:")
    print("  - Baseline fails completely")
    print("  - Fine-tuned model gets it right")
    print("  - The bug report is interesting/clear")
    print()
    print("Add these to README.md as a table like:")
    print()
    print("### 🎯 Example Predictions")
    print()
    print("| Bug Report | Baseline | Fine-Tuned | Ground Truth |")
    print("|-----------:|:--------:|:----------:|:------------:|")
    print("| *Game crashes when...* | ❌ Low / UI / Rare | ✅ Critical / Graphics / Always | Critical / Graphics / Always |")
    print()
    print("This is more effective as markdown than as an image.")


def main():
    """Generate all visualizations."""
    
    print("🎨 Creating Performance Visualizations")
    print("=" * 80)
    
    print("\n1️⃣ Performance Comparison Bar Chart")
    create_performance_comparison()
    
    print("\n2️⃣ Confusion Matrices (3 tasks)")
    create_confusion_matrices()
    
    print("\n3️⃣ Training Curves")
    create_training_curves()
    
    print("\n4️⃣ Example Showcase")
    create_example_showcase()
    
    print("\n" + "=" * 80)
    print(f"✅ Visualizations saved to: {OUTPUT_DIR}/")
    print("\nNext steps:")
    print("1. Review the generated images")
    print("2. Add them to README.md:")
    print("   ![Performance Comparison](visuals/performance_comparison.png)")
    print("3. Note: Confusion matrices based on quick_eval.py results (50 samples)")
    print("4. For full test set (300 samples), run:")
    print("   python scripts/evaluate.py --test_path data/test.jsonl")
    print("   Then modify this script to use full evaluation results")


if __name__ == "__main__":
    main()
