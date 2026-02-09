"""
Quick Evaluation Script for Hyperparameter Comparison
Use this for rapid evaluation (50 samples) during hyperparameter search
"""

import os
import sys
import argparse

# Run quick evaluation
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to model (e.g., outputs/final_model_r4)")
    parser.add_argument("--config_name", type=str, default="", help="Config name for logging (e.g., V2-r4)")
    args = parser.parse_args()
    
    # Run evaluate.py with 50 samples
    cmd = f"python /Users/faraz/Projects/Uni/7375/LLM\\ Fine\\ Tuning/scripts/evaluate.py --model_path '{args.model_path}' --num_samples 50"
    
    print(f"\n{'='*70}")
    print(f"🔍 Quick Evaluation: {args.config_name if args.config_name else args.model_path}")
    print(f"{'='*70}")
    print(f"Samples: 50 (±14% margin at 95% confidence)")
    print(f"Command: {cmd}")
    print(f"{'='*70}\n")
    
    os.system(cmd)

if __name__ == "__main__":
    main()
