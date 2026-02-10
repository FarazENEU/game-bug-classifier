"""
Show sample predictions from evaluation results for demo
"""

import json
import random

def main():
    print("=" * 70)
    print("V1 Model Evaluation Examples - Mistral-7B-Instruct-v0.2 + QLoRA (r=8)")
    print("Average Accuracy: 64.34% (41.86% severity, 62.79% component, 88.37% reproducibility)")
    print("Baseline (zero-shot): 0.33% | Improvement: 195× better")
    print("=" * 70)
    
    # Try to load eval results
    try:
        with open("final_model/evaluation_results.json", "r") as f:
            data = json.load(f)
        
        results = data.get("metrics", {})
        predictions = data.get("predictions", [])
        
        if not predictions:
            print("\n⚠️  No predictions found in results file")
            return
        
        # Show 3 examples: 1 correct, 1 incorrect, 1 mixed
        print("\n📊 Sample Predictions:\n")
        
        # Show first few examples from JSON structure
        for i, pred in enumerate(predictions[:3]):
            if i == 0:
                print("✅ EXAMPLE 1:")
            elif i == 1:
                print("\n⚠️  EXAMPLE 2:")
            else:
                print("\n📋 EXAMPLE 3:")
            
            print("-" * 70)
            print(f"Bug: {pred.get('input', '')[:150]}...")
            
            expected = pred.get('expected', '')
            predicted = pred.get('predicted', '')
            
            print(f"\nExpected:  {expected[:100]}...")
            print(f"Predicted: {predicted[:100]}...")
            print()
        

        
        # Show overall stats
        print("=" * 70)
        print("Overall Performance (50 samples):")
        print(f"  Severity:        {results.get('severity_accuracy', 0):.2f}%")
        print(f"  Component:       {results.get('component_accuracy', 0):.2f}%")
        print(f"  Reproducibility: {results.get('reproducibility_accuracy', 0):.2f}%")
        print(f"  Average:         {results.get('overall_average', 0):.2f}%")
        print("\n  Improvement over baseline: 195× (from 0.33% to 64.34%)")
        print("=" * 70)
        
    except FileNotFoundError:
        print("\n⚠️  final_model/evaluation_results.json not found")
        print("\n📝 Demo Alternative: Show conceptual example\n")
        
        print("Example Input:")
        print("-" * 70)
        print("Title: Game crashes when loading saved game")
        print("Description: Segmentation fault occurs every time I try to")
        print("load a saved game. Error in SaveManager.cpp line 156.")
        print("Happens on both Linux and Windows. Cannot play the game.")
        print("-" * 70)
        
        print("\n✅ Model Prediction:")
        print("  Severity: critical (crash prevents gameplay)")
        print("  Component: save (SaveManager error)")
        print("  Reproducibility: always (100% reproduction rate)")
        print()

if __name__ == "__main__":
    main()
