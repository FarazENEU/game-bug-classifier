# Label Improvement & Retraining Guide

## Current Status
- Initial model trained with keyword-based labels
- **Evaluation Results:**
  - Severity: 41.86% ❌ (terrible)
  - Component: 62.79% ⚠️ (mediocre)
  - Reproducibility: 88.37% ✅ (good)
  - Overall: 64.34%

**Problem:** Keyword-based labels are low quality. Model can't learn better than training data.

**Solution:** Use Mistral base model (instruction-tuned, smarter than keywords) to re-label training data, then retrain.

---

## Step 1: Prepare Data on Kaggle

### Upload train.jsonl as Kaggle Dataset:
1. Go to Kaggle → Create New Dataset
2. Name: `game-bug-reports-training`
3. Upload: `data/splits/train.jsonl` (1399 examples)
4. Make it private
5. Note the dataset path: `/kaggle/input/game-bug-reports-training/train.jsonl`

---

## Step 2: Run Label Improvement on Kaggle

### Create new notebook cell or new notebook:

```python
# Install dependencies (if needed)
!pip install -q transformers==4.46.3 peft==0.13.2 bitsandbytes==0.45.0

# Upload improve_labels.py to Kaggle, then run:
!python improve_labels.py
```

**What it does:**
- Loads Mistral-7B-Instruct base model (cached on Kaggle)
- Uses 4-bit quantization to save memory
- Re-labels all 1399 training examples with better reasoning
- Outputs: `train_improved.jsonl`

**Time:** ~30-40 minutes
**Cost:** FREE (Kaggle GPU)

---

## Step 3: Retrain with Improved Labels

### Modify train.py to use improved labels:

In the train.py notebook, change the data loading:

```python
# OLD:
train_dataset = load_dataset('json', data_files={'train': './train.jsonl'})

# NEW:
train_dataset = load_dataset('json', data_files={'train': './train_improved.jsonl'})
```

Then run the full training script again:

```bash
!python train.py
```

**Time:** ~3-4 hours (same as before)
**Expected Results:** Severity accuracy should jump from 41% → 60-70%+

---

## Step 4: Download & Re-evaluate

Download trained model:
```bash
!zip -r bug_classifier_model_v2.zip /kaggle/working/outputs/final_model
```

Re-run evaluation:
```bash
!python evaluate.py \
    --model_path /kaggle/working/outputs/final_model \
    --base_model mistralai/Mistral-7B-Instruct-v0.2 \
    --test_path /kaggle/input/data-splits/test.jsonl \
    --num_samples 50
```

**Expected improvement:**
- Severity: 41% → 65-70%
- Component: 62% → 70-75%
- Reproducibility: 88% → 90%+
- **Overall: 64% → 75%+**

---

## Why This Works (Bootstrapping)

**Problem with keyword labels:**
- "crash" → critical (too simple)
- Misses context: "crash fixed" vs "crash occurs"
- Can't understand nuance

**Mistral base model:**
- Instruction-tuned on billions of tokens
- Understands context and reasoning
- Can classify based on impact, not just keywords
- Creates higher-quality training signal

**Result:**
- Better labels → Better fine-tuned model
- Model learns from smarter teacher (base model)
- Bootstrapping effect lifts quality

---

## Timeline (28 hours remaining)

**Tonight (8 PM - 1 AM):** 5 hours
- 9 PM: Upload data, run improve_labels.py (40 min)
- 10 PM: Start retraining
- 1 AM: Training ~75% done, go to sleep

**Tomorrow morning (8 AM):** 
- Training complete, download model
- Run evaluation, verify 70%+ accuracy
- Build Flask API demo (2 hours)

**Tomorrow afternoon (12 PM - 6 PM):**
- Write business case (1 hour)
- Record video walkthrough (2 hours)
- Write technical report (2 hours)

**Tomorrow evening (6 PM - 11:59 PM):**
- Final polish and submission
- Buffer time for fixes

---

## Success Criteria

**Minimum (15/20 quality score):**
- Model accuracy ~60-65%
- Working evaluation
- Basic documentation

**Target (20/20 quality score):**
- Model accuracy ~70-75%
- Comparative analysis vs baseline
- Flask API demo
- Business case with ROI
- Professional video
- Comprehensive technical report

We're going for 20/20! 🚀
