"""
Run evaluations for all three hyperparameter configurations and compare results
Run this AFTER all training completes
"""

import subprocess
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
    print("📊 EVALUATION & COMPARISON WORKFLOW")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis will:")
    print("  1. Evaluate V2-r4 (100 samples) - ~20 mins")
    print("  2. Evaluate V2-r8 (100 samples) - ~20 mins")
    print("  3. Evaluate V2-r16 (100 samples) - ~20 mins")
    print("  4. Compare all results - <1 min")
    print("\nTotal estimated time: ~1 hour")
    print("="*70)
    print("\n🚀 Starting evaluations...\n")
    
    experiments = [
        {
            'name': 'V2-r4',
            'model_path': '/kaggle/working/outputs_r4/final_model',
            'result_file': 'evaluation_results_v2_r4.json'
        },
        {
            'name': 'V2-r8',
            'model_path': '/kaggle/working/outputs_r8/final_model',
            'result_file': 'evaluation_results_v2_r8.json'
        },
        {
            'name': 'V2-r16',
            'model_path': '/kaggle/working/outputs_r16/final_model',
            'result_file': 'evaluation_results_v2_r16.json'
        }
    ]
    
    results = []
    overall_start = time.time()
    
    # Run evaluations
    for exp in experiments:
        print(f"\n\n{'='*70}")
        print(f"📊 EVALUATING: {exp['name']}")
        print(f"{'='*70}\n")
        
        # Check if model exists
        if not os.path.exists(exp['model_path']):
            print(f"⚠️ Model not found: {exp['model_path']}")
            print(f"Skipping {exp['name']} evaluation")
            results.append({'name': exp['name'], 'status': 'MODEL_NOT_FOUND'})
            continue
        
        # Evaluate
        eval_cmd = f"""python scripts/evaluate.py \
            --model_path {exp['model_path']} \
            --num_samples 100"""
        
        success = run_command(eval_cmd, f"Evaluating {exp['name']}")
        
        if success:
            # Save with unique name
            save_cmd = f"cp evaluation_results.json {exp['result_file']}"
            run_command(save_cmd, f"Saving {exp['name']} results")
            
            print(f"\n💾 {exp['name']} results saved to: {exp['result_file']}")
            results.append({'name': exp['name'], 'status': 'SUCCESS'})
        else:
            results.append({'name': exp['name'], 'status': 'EVAL_FAILED'})
    
    # Compare results
    print("\n\n" + "="*70)
    print("📈 COMPARING RESULTS")
    print("="*70)
    
    compare_cmd = "python scripts/compare_results.py"
    compare_success = run_command(compare_cmd, "Generating comparison analysis")
    
    # Summary
    total_time = time.time() - overall_start
    print("\n\n" + "="*70)
    print("🎉 EVALUATION & COMPARISON COMPLETE!")
    print("="*70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n📊 SUMMARY:")
    for r in results:
        status_emoji = "✅" if r['status'] == 'SUCCESS' else "❌" if r['status'] == 'EVAL_FAILED' else "⚠️"
        print(f"{status_emoji} {r['name']}: {r['status']}")
    
    if compare_success:
        print("\n✅ Comparison analysis generated!")
    
    print("\n\n📋 RESULTS FILES:")
    print("Individual evaluations:")
    for exp in experiments:
        print(f"  - {exp['result_file']}")
    print("\nComparison analysis:")
    print("  - Check console output above for comparison table")
    print("  - Or check compare_results.py output")
    
    print("\n\n📋 NEXT STEPS:")
    print("1. Review comparison analysis above")
    print("2. Copy markdown table to docs/HYPERPARAMETER_OPTIMIZATION.md")
    print("3. Download all result JSON files")
    print("4. Fill technical report with results")
    print("="*70)

if __name__ == "__main__":
    main()
