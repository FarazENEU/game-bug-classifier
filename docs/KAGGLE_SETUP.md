# Kaggle Training Setup Guide

## 🚀 Quick Start (15 minutes)

### Step 1: Push Code to GitHub (5 min)

```bash
cd "/Users/faraz/Projects/Uni/7375/LLM Fine Tuning"

# Initialize git
git init
git add .
git commit -m "Initial commit: Game bug classifier project"

# Create repo on GitHub (do this in browser):
# https://github.com/new
# Name: game-bug-classifier
# Keep it public (easier for Kaggle)

# Push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/game-bug-classifier.git
git branch -M main
git push -u origin main
```

### Step 2: Setup Kaggle (5 min)

1. **Go to Kaggle**: https://www.kaggle.com
2. **Create new notebook**: 
   - Click "Code" → "New Notebook"
   - Title: "Game Bug Classifier Training"
3. **Enable GPU**: 
   - Right sidebar → "Accelerator" → Select "GPU P100" or "GPU T4"
   - Session options → "Internet" → ON

### Step 3: Clone & Upload Data (5 min)

In the first Kaggle cell:

```python
# Clone your GitHub repo
!git clone https://github.com/YOUR_USERNAME/game-bug-classifier.git
%cd game-bug-classifier
```

**Upload processed data:**
- Right sidebar → "Add Data" → "Upload"
- Upload these 3 files from your local machine:
  - `data/splits/train.jsonl`
  - `data/splits/val.jsonl`  
  - `data/splits/test.jsonl`
- They'll appear in `/kaggle/input/bug-data/`

### Step 4: Install Dependencies

In next cell:

```python
# Install requirements
!pip install -q transformers>=4.35.0 datasets>=2.14.0 peft>=0.6.0 bitsandbytes>=0.41.0 accelerate>=0.24.0
```

### Step 5: Start Training

Create new cell:

```python
# Adjust paths to Kaggle structure
!python scripts/train.py \
    --model_name "mistralai/Mistral-7B-Instruct-v0.2" \
    --train_path "/kaggle/input/bug-data/train.jsonl" \
    --val_path "/kaggle/input/bug-data/val.jsonl" \
    --output_dir "/kaggle/working/outputs" \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4
```

**Training Time:** 3-4 hours on P100 GPU

### Step 6: Download Trained Model

After training completes:

```python
# Zip the model
!zip -r bug_classifier_model.zip /kaggle/working/outputs/models/final_model/

# Download via Kaggle UI
# Right-click on bug_classifier_model.zip → Download
```

---

## ⚡ Alternative: Faster Smaller Model (1-2 hours)

If Mistral-7B is too slow, use GPT-2:

```python
!python scripts/train.py \
    --model_name "gpt2" \
    --train_path "/kaggle/input/bug-data/train.jsonl" \
    --val_path "/kaggle/input/bug-data/val.jsonl" \
    --output_dir "/kaggle/working/outputs" \
    --num_epochs 5 \
    --batch_size 8 \
    --learning_rate 5e-5
```

---

## 🔧 Troubleshooting

**Out of Memory?**
- Reduce `--batch_size` to 2
- Change `--model_name` to "gpt2"

**Training too slow?**
- Make sure GPU is enabled (not CPU)
- Check GPU usage: `!nvidia-smi`

**Can't access model?**
- Some models require HuggingFace login
- Add cell: `!huggingface-cli login` with your HF token

---

## 📊 Monitor Training

Training logs show in cell output. Look for:
- `train_loss` - should decrease
- `eval_loss` - should decrease  
- `learning_rate` - should decrease over time

---

## ✅ When Complete

You'll have:
- `bug_classifier_model.zip` - Download this (~200 MB)
- Training logs in output
- TensorBoard logs in `/kaggle/working/outputs/runs/`

Transfer the model back to your local machine for evaluation and demo!
