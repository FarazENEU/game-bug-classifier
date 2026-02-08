"""
Check the github-issues datasets
"""
from datasets import load_dataset

# Try the most promising candidates
candidates = [
    "Niciu/github-issues",
    "Yatoro/github-issues",
    "DoyyingFace/github-issues-doy"
]

print("🔍 Checking GitHub issue datasets in detail...\n")

for dataset_name in candidates:
    try:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print('='*60)
        
        # Load dataset
        dataset = load_dataset(dataset_name, split="train")
        
        print(f"✅ Loaded successfully!")
        print(f"   Size: {len(dataset)} examples")
        print(f"   Features: {list(dataset.features.keys())}")
        
        # Show first example
        if len(dataset) > 0:
            print(f"\n   First example:")
            first = dataset[0]
            for key, value in first.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"     {key}: {value[:100]}...")
                else:
                    print(f"     {key}: {value}")
        
        print(f"\n   ✅ This dataset looks usable!")
        break  # Use the first one that works
        
    except Exception as e:
        print(f"❌ Error loading {dataset_name}: {str(e)[:100]}")

print("\n" + "="*60)
print("If none work, we'll quickly scrape our own data.")
