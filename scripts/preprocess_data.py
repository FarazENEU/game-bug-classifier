"""
Data Preprocessing Pipeline for Bug Reports
Cleans, labels, and formats bug reports for fine-tuning
"""

import pandas as pd
import json
import re
from sklearn.model_selection import train_test_split
import yaml
from pathlib import Path

# Load configuration
with open('configs/data_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def clean_text(text):
    """Clean and normalize text"""
    if not isinstance(text, str):
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove URLs (optional - keeping them might be useful)
    # text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Normalize newlines
    text = text.replace('\\n', '\n').replace('\\r', '')
    
    return text.strip()

def infer_severity(title, body, labels):
    """
    Infer bug severity from title, body, and labels
    Returns: critical, high, medium, or low
    """
    text = f"{title} {body} {' '.join(labels)}".lower()
    
    # Critical: crashes, data loss, security
    if any(word in text for word in ['crash', 'critical', 'blocker', 'data loss', 'security', 'segfault', 'corruption']):
        return 'critical'
    
    # High: major functionality broken, regression
    if any(word in text for word in ['regression', 'broken', 'major', 'severe', 'high priority', "doesn't work"]):
        return 'high'
    
    # Low: cosmetic, trivial, minor UI
    if any(word in text for word in ['typo', 'cosmetic', 'minor', 'trivial', 'low priority', 'enhancement']):
        return 'low'
    
    # Default: medium
    return 'medium'

def infer_component(title, body, labels):
    """
    Infer affected component from title, body, and labels
    Returns: ui, gameplay, audio, graphics, network, performance, save, or other
    """
    text = f"{title} {body} {' '.join(labels)}".lower()
    
    # Check each component
    if any(word in text for word in ['ui', 'gui', 'interface', 'menu', 'hud', 'display', 'button', 'window']):
        return 'ui'
    
    if any(word in text for word in ['gameplay', 'mechanic', 'logic', 'control', 'input', 'player', 'game']):
        return 'gameplay'
    
    if any(word in text for word in ['audio', 'sound', 'music', 'sfx', 'volume', 'speaker']):
        return 'audio'
    
    if any(word in text for word in ['graphic', 'render', 'visual', 'shader', 'texture', 'material', 'lighting', 'fps']):
        return 'graphics'
    
    if any(word in text for word in ['network', 'multiplayer', 'online', 'connection', 'server', 'client', 'sync']):
        return 'network'
    
    if any(word in text for word in ['performance', 'lag', 'slow', 'optimization', 'memory', 'cpu', 'leak']):
        return 'performance'
    
    if any(word in text for word in ['save', 'load', 'persistence', 'data', 'file', 'export', 'import']):
        return 'save'
    
    return 'other'

def infer_reproducibility(body):
    """
    Infer how reproducible the bug is
    Returns: always, sometimes, or rare
    """
    if not body:
        return 'sometimes'
    
    text = body.lower()
    
    if any(word in text for word in ['always', '100%', 'consistent', 'every time', 'reproducible']):
        return 'always'
    
    if any(word in text for word in ['rare', 'once', 'random', 'hard to reproduce', 'occasionally']):
        return 'rare'
    
    return 'sometimes'

def format_for_training(row):
    """
    Format bug report for instruction-tuning
    """
    # Create input text
    input_text = f"Title: {row['title']}\n\nDescription: {row['body'][:800]}"
    
    # Create expected output
    output_text = f"""Severity: {row['severity']}
Component: {row['component']}
Reproducibility: {row['reproducibility']}
Summary: {row['title']}"""
    
    return {
        'instruction': 'Classify this bug report and provide a structured analysis.',
        'input': input_text,
        'output': output_text,
        'repository': row['repository'],
        'original_state': row['state']
    }

def main():
    print("="*60)
    print("🔧 Bug Report Preprocessing Pipeline")
    print("="*60)
    
    # Load raw data
    print("\n📂 Loading raw data...")
    df = pd.read_csv('data/raw/game_bug_reports.csv')
    print(f"   Loaded {len(df)} bug reports")
    
    # 1. Data Cleaning
    print("\n🧹 Cleaning data...")
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['title', 'body'])
    print(f"   After deduplication: {len(df)} reports")
    
    # Remove reports with missing or very short bodies
    df = df[df['body'].notna()]
    df = df[df['body'].str.len() >= 50]
    print(f"   After filtering short reports: {len(df)} reports")
    
    # Clean text
    df['title'] = df['title'].apply(clean_text)
    df['body'] = df['body'].apply(clean_text)
    
    # Parse labels (they're stored as string representations of lists)
    df['labels'] = df['labels'].apply(lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else [])
    
    # 2. Label Extraction
    print("\n🏷️  Extracting labels...")
    df['severity'] = df.apply(lambda row: infer_severity(row['title'], row['body'], row['labels']), axis=1)
    df['component'] = df.apply(lambda row: infer_component(row['title'], row['body'], row['labels']), axis=1)
    df['reproducibility'] = df['body'].apply(infer_reproducibility)
    
    # Print label distribution
    print(f"\n   Severity distribution:")
    print(f"      {df['severity'].value_counts().to_dict()}")
    print(f"\n   Component distribution:")
    print(f"      {df['component'].value_counts().to_dict()}")
    print(f"\n   Reproducibility distribution:")
    print(f"      {df['reproducibility'].value_counts().to_dict()}")
    
    # 3. Format for training
    print("\n📝 Formatting for instruction-tuning...")
    formatted_data = df.apply(format_for_training, axis=1).tolist()
    
    # 4. Train/Val/Test Split
    print("\n📊 Splitting data...")
    
    # First split: train+val vs test
    train_val, test = train_test_split(
        formatted_data, 
        test_size=0.15, 
        random_state=42,
        stratify=[d['output'].split('\n')[0] for d in formatted_data]  # Stratify by severity
    )
    
    # Second split: train vs val
    train, val = train_test_split(
        train_val,
        test_size=0.176,  # 0.176 of 0.85 ≈ 0.15 of total
        random_state=42,
        stratify=[d['output'].split('\n')[0] for d in train_val]
    )
    
    print(f"   Train: {len(train)} examples ({len(train)/len(formatted_data)*100:.1f}%)")
    print(f"   Val: {len(val)} examples ({len(val)/len(formatted_data)*100:.1f}%)")
    print(f"   Test: {len(test)} examples ({len(test)/len(formatted_data)*100:.1f}%)")
    
    # 5. Save processed data
    print("\n💾 Saving processed data...")
    
    # Create directories
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    Path('data/splits').mkdir(parents=True, exist_ok=True)
    
    # Save full processed dataset
    with open('data/processed/formatted_bugs.json', 'w') as f:
        json.dump(formatted_data, f, indent=2)
    
    # Save splits
    with open('data/splits/train.jsonl', 'w') as f:
        for item in train:
            f.write(json.dumps(item) + '\n')
    
    with open('data/splits/val.jsonl', 'w') as f:
        for item in val:
            f.write(json.dumps(item) + '\n')
    
    with open('data/splits/test.jsonl', 'w') as f:
        for item in test:
            f.write(json.dumps(item) + '\n')
    
    print(f"\n✅ Preprocessing complete!")
    print(f"   Processed data saved to: data/processed/")
    print(f"   Train/val/test splits saved to: data/splits/")
    
    # Show example
    print(f"\n📄 Example formatted bug report:")
    print("="*60)
    example = train[0]
    print(f"Instruction: {example['instruction']}")
    print(f"\nInput:\n{example['input'][:300]}...")
    print(f"\nOutput:\n{example['output']}")
    print("="*60)

if __name__ == "__main__":
    main()
