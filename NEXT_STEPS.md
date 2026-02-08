# 🎯 NEXT STEPS - Complete Your Assignment

## ⏰ Time Remaining: ~9 hours of work

---

## 🚀 Step 1: Push to GitHub (5 minutes - DO THIS NOW)

```bash
cd "/Users/faraz/Projects/Uni/7375/LLM Fine Tuning"

# Initialize git
git init
git add .
git commit -m "Complete bug classifier project setup"

# Create repo on GitHub:
# 1. Go to: https://github.com/new
# 2. Name: game-bug-classifier
# 3. Keep PUBLIC (easier for Kaggle)
# 4. Don't add README (we have one)

# Link and push (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/game-bug-classifier.git
git branch -M main
git push -u origin main
```

---

## 🤖 Step 2: Train on Kaggle (4 hours - Can run overnight)

### Follow the guide: [`docs/KAGGLE_SETUP.md`](docs/KAGGLE_SETUP.md)

**Quick summary:**
1. Go to kaggle.com → Create notebook
2. Enable GPU (P100 or T4) - RIGHT SIDEBAR
3. Clone your GitHub repo in first cell
4. Upload 3 data files:
   - `data/splits/train.jsonl`
   - `data/splits/val.jsonl`
   - `data/splits/test.jsonl`
5. Install dependencies: `!pip install -q transformers datasets peft bitsandbytes accelerate`
6. Run training: `!python scripts/train.py --train_path /kaggle/input/bug-data/train.jsonl --val_path /kaggle/input/bug-data/val.jsonl`
7. Download model when done (~3-4 hours)

**⚡ Alternative: Faster Training (1-2 hours)**
Use GPT-2 instead of Mistral-7B:
```bash
!python scripts/train.py \
    --model_name "gpt2" \
    --train_path /kaggle/input/bug-data/train.jsonl \
    --val_path /kaggle/input/bug-data/val.jsonl \
    --num_epochs 5 \
    --batch_size 8
```

---

## 📊 Step 3: Evaluate Model (30 minutes)

After downloading the model from Kaggle:

```bash
# Extract model
unzip bug_classifier_model.zip -d outputs/

# Run evaluation
python scripts/evaluate.py \
    --model_path outputs/models/final_model \
    --test_path data/splits/test.jsonl
```

This gives you:
- Severity accuracy
- Component accuracy  
- Reproducibility accuracy
- Sample predictions
- `evaluation_results.json` file

---

## 🎮 Step 4: Test Interactive Demo (20 minutes)

```bash
# Run demo
python scripts/demo.py --model_path outputs/models/final_model
```

Try the 3 built-in examples, then create custom bug reports!

---

## 🎥 Step 5: Record Video (30-60 minutes)

**Record a 5-10 minute walkthrough showing:**

1. **Introduction** (1 min)
   - Project goal
   - Why game bug classification?

2. **Data** (2 min)
   - Show `data/raw/game_bug_reports.csv` 
   - Explain collection from 4 game repos
   - Show split statistics

3. **Model & Training** (2 min)
   - Show `configs/model_config.yaml`
   - Explain LoRA + 4-bit quantization
   - Show Kaggle training logs/screenshots

4. **Results** (2 min)
   - Show `evaluation_results.json`
   - Discuss accuracy metrics
   - Show error analysis

5. **Live Demo** (2-3 min)
   - Run `scripts/demo.py`
   - Classify 2-3 bug reports live
   - Show it works!

**Tools for Recording:**
- macOS: QuickTime Player (File → New Screen Recording)
- Free: OBS Studio

---

## 📝 Step 6: Write Report (1-2 hours)

Create `docs/TECHNICAL_REPORT.md` covering:

### 1. Introduction & Motivation
- Why this problem matters
- Use cases in game development

### 2. Data Collection & Preprocessing
- GitHub repos selected
- Filtering criteria
- Label inference methodology
- Statistics (2000 reports → 1998 clean)

### 3. Model Architecture
- Base model: Mistral-7B-Instruct-v0.2
- Fine-tuning: LoRA (r=8, alpha=32)
- Quantization: 4-bit (nf4)
- Why these choices?

### 4. Training Process
- Hyperparameters
- Training time & resources
- Loss curves (if available)

### 5. Evaluation Results
- Test set metrics
- Confusion matrices
- Error analysis
- What worked? What didn't?

### 6. Demo & Applications
- How to use the system
- Example predictions
- Future improvements

### 7. Conclusions
- Lessons learned
- Next steps

---

## ✅ Step 7: Final Submission Checklist

- [ ] Code on GitHub (public repo)
- [ ] All scripts tested and working
- [ ] `evaluation_results.json` generated
- [ ] Video recorded and uploaded (YouTube/Google Drive)
- [ ] Technical report written
- [ ] README updated with results
- [ ] Submit:
  - GitHub repo link
  - Video link
  - Technical report PDF
  - Model files (if required)

---

## 💡 Tips for Success

**For the Assignment:**
- Show your work! Document everything
- Include screenshots of training
- Discuss what you learned from failures
- Quality > Quantity in video

**If You're Short on Time:**
- Use GPT-2 instead of Mistral (trains in 1-2 hours)
- Test on 100 samples instead of all 300
- Focus on one classification task (severity only)
- Keep video under 8 minutes

**Common Issues:**
- **Kaggle GPU quota exceeded**: Wait or use different account
- **Model too large**: Use GPT-2 or distilgpt2
- **Out of memory**: Reduce batch_size to 2
- **Slow download**: Use Kaggle CLI instead of browser

---

## 📞 Quick Reference

- Training script: `scripts/train.py`
- Evaluation: `scripts/evaluate.py`
- Demo: `scripts/demo.py`
- Kaggle guide: `docs/KAGGLE_SETUP.md`
- Full setup: `docs/SETUP.md`

**You've got this! 🚀**
