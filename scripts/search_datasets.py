"""
Script to search for suitable bug report datasets on Hugging Face
"""
from datasets import list_datasets

print("🔍 Searching for bug report datasets on Hugging Face...\n")

# Search for relevant keywords
keywords = ['bug', 'issue', 'github', 'software']

print("Searching datasets with keywords:  bug, issue, github, software\n")

# Get all datasets
all_datasets = list_datasets()

# Filter for relevant ones
relevant = []
for dataset_id in all_datasets:
    dataset_lower = dataset_id.lower()
    if any(keyword in dataset_lower for keyword in keywords):
        relevant.append(dataset_id)

print(f"Found {len(relevant)} potentially relevant datasets:\n")
for i, dataset in enumerate(relevant[:20], 1):  # Show first 20
    print(f"{i}. {dataset}")

print("\n📊 Recommended datasets for bug classification:")
print("1. github-issues (if available)")
print("2. mozilla-foundation/common_voice")  
print("3. Search manually on: https://huggingface.co/datasets?search=github+bugs")
