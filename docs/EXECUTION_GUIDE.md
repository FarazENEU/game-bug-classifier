# Hyperparameter Optimization - Execution Guide

## 🎯 Goal
Train and evaluate 3 LoRA rank configurations to find optimal balance between expressiveness and overfitting.

**Configurations to Test:**
- **V2-r4:** r=4, α=16 (8M trainable params - lower capacity)
- **V2-r8:** r=8, α=32 (16M trainable params - baseline)
- **V2-r16:** r=16, α=64 (32M trainable params - higher capacity)

**Note:** All three configurations will be trained sequentially in one automated run!

## 📋 Prerequisites
- ✅ train_improved.jsonl uploaded to Kaggle dataset (data-splits)
- ✅ val.jsonl uploaded to Kaggle dataset (data-splits)
- ✅ test.jsonl uploaded to Kaggle dataset (data-splits)
- ✅ 2× Tesla T4 GPUs available

## ⏱️ Time Estimate
- Zero-shot baseline eval (100 samples): 30 mins **← Required by rubric!**
- ALL THREE configs training + eval: 7 hours (automated!)
- Analysis + documentation: 30 mins
- **Total: ~8 hours**

**Evaluation Strategy:** Use 100 samples for ALL evaluations (baseline + 3 configs) for consistent comparison, ±10% margin at 95% confidence. We're comparing relative performance, not measuring final production accuracy.

---

## Step-by-Step Instructions

### Step 0: Zero-Shot Baseline Evaluation (IMPORTANT - 4 rubric points!)

**Purpose:** Compare fine-tuned model against Mistral base model with no training (required by rubric).

**Before starting hyperparameter experiments, run this once:**

```python
# In new Kaggle cell - uses base Mistral without any fine-tuning
# This evaluates zero-shot performance for comparison

!python scripts/evaluate_baseline.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --test_path /kaggle/input/data-splits/test.jsonl \
    --num_samples 100 \
    --output_file baseline_zero_shot_results.json
```

**Time:** ~30 mins (100 samples, loads base model from HF)

**Expected results:** Likely 30-40% overall accuracy (unstructured output, inconsistent format)
**This establishes why fine-tuning matters!**

**Download:** baseline_zero_shot_results.json

---

### Step 1: Train ALL THREE Configurations (AUTOMATED!)

**🚀 EASY METHOD - Run both experiments sequentially:**

Upload `scripts/run_hyperparameter_experiments.py` to Kaggle, then:

```python
# This runs BOTH r=4 and r=16 experiments automatically!
# Trains, evaluates, zips, and saves results for both configs
!python scripts/run_hyperparameter_experiments.py
```

**Time:** ~4.5 hours total (hands-off!)

**What it does:**
1. Trains V2-r4 (r=4, α=16) → 2 hours
2. Evaluates V2-r4 (100 samples) → 20 mins
3. Trains V2-r16 (r=16, α=64) → 2 hours
4. Evaluates V2-r16 (100 samples) → 20 mins
5. Shows progress and summary

**Don't forget to download after completion:**
- bug_classifier_v2_r4.zip
- bug_classifier_v2_r16.zip
- evaluation_results_v2_r4.json
- evaluation_results_v2_r16.json

---

**📝 MANUAL METHOD (if automated script fails):**

<details>
<summary>Click to expand manual commands</summary>

### Step 2: Train V2-r4 (Low Rank)

```python
# V2-r4: Lower capacity
!python train.py \
    --train_path /kaggle/working/train_improved.jsonl \
    --val_path /kaggle/input/data-splits/val.jsonl \
    --output_dir /kaggle/working/outputs_r4 \
    --lora_r 4 \
    --lora_alpha 16 \
    --num_epochs 3 \
    -ONE SCRIPT RUNS ALL THREE:**

Upload `scripts/run_hyperparameter_experiments.py` to Kaggle, then:

```python
# This runs ALL THREE experiments (r=4, r=8, r=16) sequentially!
# Completely hands-off for ~7 hours
!python scripts/run_hyperparameter_experiments.py
```

**Time:** ~7 hours total (fully automated!)

**What it does:**
1. Trains V2-r4 (r=4, α=16) → 2 hours
2. Evaluates V2-r4 (100 samples) → 20 mins
3. Trains V2-r8 (r=8, α=32) → 2 hours
4. Evaluates V2-r8 (100 samples) → 20 mins
5. Trains V2-r16 (r=16, α=64) → 2 hours
6. Evaluates V2-r16 (100 samples) → 20 mins
7. Zips all models automatically
8. Saves all results with clear names
9. Shows summary comparison at the end

**⚠️ CRITICAL:** Keep Kaggle notebook active! Click around every 45-50 mins to prevent 60-min timeout.

**Don't forget to download after completion:**
- bug_classifier_v2_r4.zip
- bug_classifier_v2_r8.zip
- bug_classifier_v2_r16.zip
- evaluation_results_v2_r4.json
- evaluation_results_v2_r8
    --max_length 512
Run each configuration separately with these commands:

```python
# V2-r4: Lower capacity
!python train.py --train_path /kaggle/input/data-splits/train_improved.jsonl \
    --val_path /kaggle/input/data-splits/val.jsonl --output_dir /kaggle/working/outputs_r4 \
    --lora_r 4 --lora_alpha 16 --num_epochs 3 --batch_size 4 --learning_rate 2e-4 --max_length 512
!cd /kaggle/working && zip -r bug_classifier_v2_r4.zip outputs_r4/final_model/
!python evaluate.py --model_path /kaggle/working/outputs_r4/final_model --num_samples 100
!cp evaluation_results.json evaluation_results_v2_r4.json

# V2-r8: Baseline
!python train.py --train_path /kaggle/input/data-splits/train_improved.jsonl \
    --val_path /kaggle/input/data-splits/val.jsonl --output_dir /kaggle/working/outputs_r8 \
    --lora_r 8 --lora_alpha 32 --num_epochs 3 --batch_size 4 --learning_rate 2e-4 --max_length 512
!cd /kaggle/working && zip -r bug_classifier_v2_r8.zip outputs_r8/final_model/
!python evaluate.py --model_path /kaggle/working/outputs_r8/final_model --num_samples 100
!cp evaluation_results.json evaluation_results_v2_r8.json

# V2-r16: Higher capacity
!python train.py --train_path /kaggle/input/data-splits/train_improved.jsonl \
    --val_path /kaggle/input/data-splits/val.jsonl --output_dir /kaggle/working/outputs_r16 \
    --lora_r 16 --lora_alpha 64 --num_epochs 3 --batch_size 4 --learning_rate 2e-4 --max_length 512

**📝 MANUAL METHOD (if needed):**

<details>
<summary>Click to expand manual analysis</summary>

**Note:** All evaluated on same 100 samples for fair comparison (±10% margin)

**Open docs/HYPERPARAMETER_OPTIMIZATION.md and fill:**

### Results Table

| Config           | Severity | Component | Repro  | Overall | Train Loss | Params |
| ---------------- | -------- | --------- | ------ | ------- | ---------- | ------ |
| Zero-shot (base) | __%      | __%       | __%    | __%     | N/A        | 0      |
| V1 (keyword)     | 41.86%   | 62.79%    | 88.37% | 64.34%  | 1.3216     | 16M    |
| V2-r4            | __%      | __%       | __%    | __%     | __         | 8M     |
| V2-r8            | __%      | __%       | __%    | __%     | __         | 16M    |
| V2-r16           | __%      | __%       | __%    | __%     | __         | 32M    |

### Key Findings to Document

1. **Best overall accuracy:** r=__ achieved __%
2. **Severity improvements:** 
   - V2-r4: __% (vs V1 41.86%, +__%)
   - V2-r8: __% (vs V1 41.86%, +__%)
   - V2-r16: __% (vs V1 41.86%, +__%)
3. **Overfitting analysis:**
   - Did training loss decrease while eval stayed flat?
   - Is r=16 worse than r=8 despite 2× parameters?
4. **Recommendation:** Select r=__ for final model because __

</details>

---

## Step 5: Update Technical Report

**In TECHNICAL_REPORT_CONTENT.md, add section:**

### 4.2 Hyperparameter Optimization

**Strate3y (3 points):** We systematically evaluated LoRA rank (r=4, 8, 16) to balance model expressiveness against overfitting risk on our 1399-example dataset. Rank determines trainable parameters (8M→16M→32M) and represents a fundamental capacity tradeoff. We held learning rate (2e-4), batch size (16 effective), and epochs (3) constant based on literature recommendations.

**Configurations Tested (4 points):** 
[PASTE TABLE FROM ABOVE]

**Analysis (3 points):** Results revealed r=__ optimal, achieving __% overall accuracy (__% severity). [Explain why: e.g., "r=4 underfit on multi-task learning", "r=16 overfit shown by train-eval gap", "r=8 balanced"]. Training loss comparison: [describe patterns]. Selected r=__ for production based on accuracy-efficiency tradeoff and 13GB GPU memory constraint. This demonstrates parameter-efficient fine-tuning achieves competitive results with <1% of model parameters trainable.

---

## 🚨 Critical Reminders

1. **ALWAYS download model zip immediately after training completes**
2. **Don't close Kaggle notebook until files downloaded**
3. **60-min inactivity timeout** - stay active during training
4. **Save evaluation results** with descriptive names (include r=X)
5. **Record training loss** from final epoch for overfitting analysis

---

## 📊 Expected Timeline

**From now:**
+0h:           Zero-shot baseline eval (100 samples) - 30 mins
+0.5h:         V2-r8 eval (100 samples) - 20 mins
+1h:           Start V2-r4 training
+3h:           V2-r4 done, eval (100 samples), start V2-r16
+5h:           V2-r16 done, eval (100 samples)
+5.5h:         Analysis & documentation
+6h:           Hyperparameter section COMPLETE
+7.5h:         Technical report written
+9h:           Video recorded
+9.5h:         Final submission ready

With 10 hours total, you have 30 mins buffer! 🎉

**Why 100 samples?** Fast enough (~20-30 mins each) with reasonable confidence (±10% margin). Most importantly: **consistent across all evaluations** for fair comparison.
30 mins) - run while uploading scripts
+0.5h:         Start automated hyperparameter experiments (7 hours)
+7.5h:         All experiments complete, download files
+8h:           Run compare_results.py, document findings (30 mins)
+8.5h:         Hyperparameter section COMPLETE
+10h:          Technical report written (1.5 hrs)
+11.5h:        Video recorded (1.5 hrs)
+12h:          Final submission ready

With deadline in ~10 hours, this is tight but doable! Start immediately! 🚀

**You've got this!** 🚀
