"""
Quick search for existing bug report datasets
"""
from datasets import load_dataset_builder
import requests

print("🔍 Checking for existing GitHub issue/bug datasets...\n")

# Known datasets to check
candidates = [
    "giganticode/github-issues",
    "github-issues-databricks",
    "codeparrot/github-issues", 
    "bigcode/github-issues",
    "se-edu/addressbook-level4",
]

print("Checking HuggingFace datasets:\n")

for dataset_name in candidates:
    try:
        builder = load_dataset_builder(dataset_name)
        print(f"✅ Found: {dataset_name}")
        print(f"   Description: {builder.info.description[:100]}...")
        print()
    except Exception as e:
        print(f"❌ Not found: {dataset_name}")

print("\n" + "="*60)
print("Alternative: Search manually on:")
print("https://huggingface.co/datasets?search=github+issues")
print("https://www.kaggle.com/datasets?search=github+bugs")
