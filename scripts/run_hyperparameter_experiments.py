"""
Run hyperparameter experiments (r=4 and r=16) sequentially
Use this AFTER V2-r8 training completes
"""

import subprocess
import json
import time
from datetime import datetime
import os

def run_command(cmd, description):
    """Run a shell command and track time"""
    print("\n" + "="*70)
    print(f"🚀 {description}")
    print("="*70)
    print(f"Command: {cmd}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=False, text=True)
        elapsed = time.time() - start_time
        print(f"\n✅ Completed in {elapsed/60:.1f} minutes")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Failed after {elapsed/60:.1f} minutes")
        print(f"Error: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("🔬 HYPERPARAMETER OPTIMIZATION EXPERIMENTS")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis will run ALL THREE configurations:")
    print("  1. Train V2-r4 (r=4, α=16) - ~2 hours")
    print("  2. Evaluate V2-r4 - ~20 mins")
    print("  3. Train V2-r8 (r=8, α=32) - ~2 hours")
    print("  4. Evaluate V2-r8 - ~20 mins")
    print("  5. Train V2-r16 (r=16, α=64) - ~2 hours")
    print("  6. Evaluate V2-r16 - ~20 mins")
    print("\nTotal estimated time: ~7 hours")
    print("\n⚠️  Keep Kaggle notebook active (60-min timeout!)")
    print("="*70)
    print("\n🚀 Starting experiments...\n")
    
    experiments = [
        {
            'name': 'V2-r4',
            'lora_r': 4,
            'lora_alpha': 16,
            'output_dir': '/kaggle/working/outputs_r4'
        },
        {
            'name': 'V2-r8',
            'lora_r': 8,
            'lora_alpha': 32,
            'output_dir': '/kaggle/working/outputs_r8'
        },
        {
            'name': 'V2-r16',
            'lora_r': 16,
            'lora_alpha': 64,
            'output_dir': '/kaggle/working/outputs_r16'
        }
    ]
    
    results = []
    overall_start = time.time()
    
    for exp in experiments:
        exp_start = time.time()
        print(f"\n\n{'='*70}")
        print(f"📊 EXPERIMENT: {exp['name']} (r={exp['lora_r']}, α={exp['lora_alpha']})")
        print(f"{'='*70}\n")
        
        # Training
        # Note: train.py should be in same directory (both uploaded to /kaggle/working/)
        train_cmd = f"""python train.py \
            --train_path /kaggle/input/data-split/train_improved.jsonl \
            --val_path /kaggle/input/data-split/val.jsonl \
            --output_dir {exp['output_dir']} \
            --lora_r {exp['lora_r']} \
            --lora_alpha {exp['lora_alpha']} \
            --num_epochs 3 \
            --batch_size 4 \
            --learning_rate 2e-4 \
            --max_length 512"""
        
        success = run_command(train_cmd, f"Training {exp['name']}")
        
        if not success:
            print(f"\n⚠️ Training failed for {exp['name']}, skipping evaluation")
            results.append({'name': exp['name'], 'status': 'FAILED', 'error': 'Training failed'})
            continue
        
        # Zip model
        zip_cmd = f"cd /kaggle/working && zip -r bug_classifier_{exp['name'].lower().replace('-', '_')}.zip {exp['output_dir'].split('/')[-1]}/final_model/"
        run_command(zip_cmd, f"Zipping {exp['name']} model")
        
        print(f"\n💾 IMPORTANT: Download bug_classifier_{exp['name'].lower().replace('-', '_')}.zip NOW!")
        time.sleep(5)  # Give time to read the message
        
        # Evaluation
        eval_cmd = f"""python evaluate.py \
            --model_path {exp['output_dir']}/final_model \
            --num_samples 100"""
        
        success = run_command(eval_cmd, f"Evaluating {exp['name']}")
        
        if success:
            # Save results with unique name
            save_cmd = f"cp evaluation_results.json evaluation_results_{exp['name'].lower().replace('-', '_')}.json"
            run_command(save_cmd, f"Saving {exp['name']} results")
            
            # Try to load and display results
            try:
                with open('evaluation_results.json', 'r') as f:
                    eval_results = json.load(f)
                    results.append({
                        'name': exp['name'],
                        'status': 'SUCCESS',
                        'metrics': eval_results.get('accuracies', {}),
                        'time': time.time() - exp_start
                    })
                    
                    print(f"\n📊 {exp['name']} Results:")
                    print(f"  Overall: {eval_results.get('accuracies', {}).get('overall', 'N/A'):.2f}%")
                    print(f"  Severity: {eval_results.get('accuracies', {}).get('severity', 'N/A'):.2f}%")
                    print(f"  Component: {eval_results.get('accuracies', {}).get('component', 'N/A'):.2f}%")
                    print(f"  Reproducibility: {eval_results.get('accuracies', {}).get('reproducibility', 'N/A'):.2f}%")
            except Exception as e:
                print(f"⚠️ Could not load results: {e}")
                results.append({'name': exp['name'], 'status': 'COMPLETED', 'note': 'Results file not readable'})
        else:
            results.append({'name': exp['name'], 'status': 'EVAL_FAILED'})
    
    # Summary
    total_time = time.time() - overall_start
    print("\n\n" + "="*70)
    print("🎉 ALL EXPERIMENTS COMPLETE!")
    print("="*70)
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n📊 SUMMARY:")
    for r in results:
        status_emoji = "✅" if r['status'] == 'SUCCESS' else "❌"
        print(f"\n{status_emoji} {r['name']}: {r['status']}")
        if 'metrics' in r:
            print(f"   Overall: {r['metrics'].get('overall', 'N/A'):.2f}%")
    
    print("\n\n📋 NEXT STEPS:")
    print("1. Download all model zips:")
    print("   - bug_classifier_v2_r4.zip")
    print("   - bug_classifier_v2_r16.zip")
    print("2. Download all evaluation results:")
    print("   - evaluation_results_v2_r4.json")
    print("   - evaluation_results_v2_r16.json")
    print("3. Run: python scripts/compare_results.py")
    print("4. Fill docs/HYPERPARAMETER_OPTIMIZATION.md with results")
    print("="*70)

if __name__ == "__main__":
    main()
