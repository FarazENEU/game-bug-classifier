"""
Run corrected hyperparameter experiments (r=4 and r=16) with proper train.jsonl
Uses 1 epoch for faster training (~2.5 hours total)
Skips r=8 since we already have V1 model
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
    print("🔬 CORRECTED HYPERPARAMETER EXPERIMENTS")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n✅ Using CORRECT dataset: train.jsonl (not train_improved.jsonl)")
    print("✅ Using 1 epoch for faster training")
    print("✅ Skipping r=8 (already have V1 model)")
    print("\nThis will run TWO configurations:")
    print("  1. Train V3-r4 (r=4, α=16) - ~40 mins")
    print("  2. Evaluate V3-r4 - ~20 mins")
    print("  3. Train V3-r16 (r=16, α=64) - ~70 mins")
    print("  4. Evaluate V3-r16 - ~20 mins")
    print("\nTotal estimated time: ~2.5 hours")
    print("\n⚠️  Keep Kaggle notebook active (60-min timeout!)")
    print("="*70)
    print("\n🚀 Starting experiments...\n")
    
    experiments = [
        {
            'name': 'V3-r4',
            'lora_r': 4,
            'lora_alpha': 16,
            'output_dir': '/kaggle/working/outputs_v3_r4'
        },
        {
            'name': 'V3-r16',
            'lora_r': 16,
            'lora_alpha': 64,
            'output_dir': '/kaggle/working/outputs_v3_r16'
        }
    ]
    
    results = []
    overall_start = time.time()
    
    for exp in experiments:
        exp_start = time.time()
        print(f"\n\n{'='*70}")
        print(f"📊 EXPERIMENT: {exp['name']} (r={exp['lora_r']}, α={exp['lora_alpha']})")
        print(f"{'='*70}\n")
        
        # Training - USING CORRECT TRAIN.JSONL
        train_cmd = f"""python scripts/train.py \
            --train_path /kaggle/input/data-split/train.jsonl \
            --val_path /kaggle/input/data-split/val.jsonl \
            --output_dir {exp['output_dir']} \
            --lora_r {exp['lora_r']} \
            --lora_alpha {exp['lora_alpha']} \
            --num_epochs 1 \
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
        eval_cmd = f"""python scripts/evaluate.py \
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
                        'metrics': eval_results.get('metrics', {}),
                        'time': time.time() - exp_start
                    })
                    
                    print(f"\n📊 {exp['name']} Results:")
                    metrics = eval_results.get('metrics', {})
                    print(f"  Overall: {metrics.get('overall_average', 'N/A'):.2f}%")
                    print(f"  Severity: {metrics.get('severity_accuracy', 'N/A'):.2f}%")
                    print(f"  Component: {metrics.get('component_accuracy', 'N/A'):.2f}%")
                    print(f"  Reproducibility: {metrics.get('reproducibility_accuracy', 'N/A'):.2f}%")
            except Exception as e:
                print(f"⚠️ Could not load results: {e}")
                results.append({'name': exp['name'], 'status': 'COMPLETED', 'note': 'Results file not readable'})
        else:
            results.append({'name': exp['name'], 'status': 'EVAL_FAILED'})
    
    # Summary
    total_time = time.time() - overall_start
    print("\n\n" + "="*70)
    print("🎉 ALL CORRECTED EXPERIMENTS COMPLETE!")
    print("="*70)
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n📊 SUMMARY:")
    for r in results:
        status_emoji = "✅" if r['status'] == 'SUCCESS' else "❌"
        print(f"  {status_emoji} {r['name']}: {r['status']}")
        if 'metrics' in r:
            print(f"      Overall: {r['metrics'].get('overall_average', 'N/A'):.2f}%")
    
    print("\n📥 FILES TO DOWNLOAD:")
    print("  1. bug_classifier_v3_r4.zip")
    print("  2. bug_classifier_v3_r16.zip")
    print("  3. evaluation_results_v3_r4.json")
    print("  4. evaluation_results_v3_r16.json")
    
    print("\n📊 COMPARISON WITH EXISTING MODELS:")
    print("  Baseline (zero-shot): 0.33%")
    print("  V1 r=8 (3 epochs, train.jsonl): 64.34%")
    print("  V2 r=4/8/16 (3 epochs, train_improved.jsonl): ~20% ❌ FAILED")
    print("  V3 r=4 (1 epoch, train.jsonl): [see above]")
    print("  V3 r=16 (1 epoch, train.jsonl): [see above]")
    
    print("\n🔬 NEXT STEPS:")
    print("  1. Download all zip files and JSON results")
    print("  2. Compare V3-r4 vs V1-r8 vs V3-r16 (hyperparameter analysis)")
    print("  3. Document V2 failure in report (train_improved label mismatch)")
    print("  4. Update technical report with all results")
    print("  5. Record video walkthrough")
    print("="*70)

if __name__ == "__main__":
    main()
