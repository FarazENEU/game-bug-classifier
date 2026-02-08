"""
Search for real existing datasets more systematically
"""
from huggingface_hub import list_datasets

print("🔍 Searching HuggingFace for bug/issue datasets...\n")

# Search for datasets with relevant keywords
keywords = ['github', 'issue', 'bug', 'software']

found_datasets = []
for keyword in keywords:
    try:
        print(f"Searching for '{keyword}'...")
        datasets = list(list_datasets(search=keyword, limit=10))
        found_datasets.extend(datasets)
    except Exception as e:
        print(f"Error searching for {keyword}: {e}")

# Remove duplicates
unique_datasets = list(set([d.id for d in found_datasets]))

print(f"\n✅ Found {len(unique_datasets)} unique datasets\n")
print("Top relevant datasets:")
for i, dataset_id in enumerate(unique_datasets[:15], 1):
    if any(kw in dataset_id.lower() for kw in ['github', 'issue', 'bug', 'code', 'software']):
        print(f"{i}. {dataset_id}")
