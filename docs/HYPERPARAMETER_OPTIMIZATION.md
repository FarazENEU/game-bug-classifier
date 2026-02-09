# Hyperparameter Optimization Strategy

## Objective
Systematically evaluate LoRA rank (r) to find optimal balance between model expressiveness and overfitting risk on 1399 training examples with improved labels.

## Hyperparameter Search Strategy

### Primary Variable: LoRA Rank (r)
**Rationale:** LoRA rank determines the number of trainable parameters and model capacity. Critical tradeoff:
- **Low rank (r=4):** Fewer parameters (8M), lower overfitting risk, may underfit
- **Medium rank (r=8):** Balanced (16M parameters), standard in literature
- **High rank (r=16):** More parameters (32M), higher expressiveness, overfitting risk

### Configurations Tested

| Config | r | α | α/r | Trainable Params | Learning Rate | Batch Size | Epochs |
|--------|---|---|-----|------------------|---------------|------------|--------|
| **V2-r4** | 4 | 16 | 4.0 | ~8M | 2e-4 | 4×4=16 | 3 |
| **V2 (baseline)** | 8 | 32 | 4.0 | ~16M | 2e-4 | 4×4=16 | 3 |
| **V2-r16** | 16 | 64 | 4.0 | ~32M | 2e-4 | 4×4=16 | 3 |

**Held Constant:**
- Learning rate: 2e-4 (LoRA standard)
- α/r ratio: 4.0 (maintains consistent scaling)
- Batch size: 16 effective
- Epochs: 3
- Training data: 1399 examples (Mistral-improved labels)
- Hardware: 2× Tesla T4 GPUs

### Alternative Configurations Considered

**Learning Rate Search:** 1e-4, 2e-4, 4e-4
- **Not chosen:** LoRA papers strongly recommend 2e-4 for 7B models; rank is more fundamental

**Epoch Search:** 2, 3, 4 epochs
- **Not chosen:** V1 showed loss plateau at ~2.3 epochs; 3 is optimal

**Batch Size Search:** 8, 16, 32
- **Not chosen:** Constrained by GPU memory (13GB limit on 2× T4)

## Evaluation Protocol

**Consistent Evaluation (All Models):**
- Sample: 100 random test examples (same for all configs)
- Time: ~20 minutes per model
- Purpose: Fair comparison for hyperparameter selection
- Margin: ±10% at 95% confidence
- **Critical:** Use SAME 100 samples across all three models for apples-to-apples comparison

**Metrics:**
- Severity accuracy (most critical for triage)
- Component accuracy
- Reproducibility accuracy
- Overall average (arithmetic mean of 3 metrics)

**Note:** We're comparing RELATIVE performance (is r=8 better than r=4?), not measuring final production accuracy. Consistent evaluation methodology more important than large sample size.

## Expected Outcomes

### Hypothesis

**V2-r4 (Low Rank):**
- **Prediction:** 70-73% overall
- **Reasoning:** May underfit on multi-task learning (3 classification tasks), but less overfitting on 1399 examples
- **Risk:** Insufficient capacity for contextual severity reasoning

**V2 (Baseline, r=8):**
- **Prediction:** 75-78% overall (target)
- **Reasoning:** Standard LoRA rank for 7B models, balanced capacity
- **Expected:** Best overall performance

**V2-r16 (High Rank):**
- **Prediction:** 73-76% overall
- **Reasoning:** More expressive but may overfit on small dataset
- **Risk:** High variance, memorization instead of generalization

### Decision Criteria

**Select best model based on:**
1. **Overall accuracy** (primary metric)
2. **Severity accuracy** (most business-critical)
3. **Training loss curve** (overfitting indicators)
4. **Inference speed** (higher rank = slower, but negligible for r=4-16)

## Training Schedule

### Execution Plan

**Parallel Strategy (Not Possible):**
- Kaggle free tier: only 1 notebook with GPU at a time
- Must train sequentially

**Sequential Strategy:**
```
T+0h:   V2 (r=8) completes training
T+0.2h: Download V2, evaluation (100 samples)
T+0.5h: Start V2-r4 training
T+2.5h: V2-r4 completes, eval (100 samples)
T+2.8h: Start V2-r16 training  
T+4.8h: V2-r16 completes, eval (100 samples)
T+5h:   Analysis & documentation
```

**Total Time:** ~5.5 hours from V2 completion

### Implementation

**Modified train.py command:**
```bash
# V2-r4
python train.py --lora_r 4 --lora_alpha 16 --output_dir outputs/final_model_r4

# V2-r16  
python train.py --lora_r 16 --lora_alpha 64 --output_dir outputs/final_model_r16
```

**Evaluation:**
```bash
python evaluate.py --model_path outputs/final_model_r4 --num_samples 100
python evaluate.py --model_path outputs/final_model --num_samples 100
python evaluate.py --model_path outputs/final_model_r16 --num_samples 100
```

**Important:** Use same 100 samples for all three models (fair comparison)

## Results

### V2-r4 (Low Rank)

**Training Loss:**
```
[TO FILL AFTER TRAINING]
Epoch 1: [loss]
Epoch 2: [loss]
Epoch 3: [loss]
Final: [loss]
```

**Quick Evaluation (50 samples):**
```
[TO FILL]
Severity:        XX.XX%
Component:       XX.XX%
Reproducibility: XX.XX%
Overall:         XX.XX%
```

**Analysis:** [TO FILL]

---

### V2 (Baseline, r=8)

**Training Loss:**
```
[TO FILL AFTER TRAINING]
Epoch 1: [loss]
Epoch 2: [loss]  
Epoch 3: [loss]
Final: [loss]
```

**Full Evaluation (300 samples):**
```
[TO FILL]
Severity:        XX.XX%
Component:       XX.XX%
Reproducibility: XX.XX%
Overall:         XX.XX%
```

**Analysis:** [TO FILL]

---

### V2-r16 (High Rank)

**Training Loss:**
```
[TO FILL AFTER TRAINING]
Epoch 1: [loss]
Epoch 2: [loss]
Epoch 3: [loss]
Final: [loss]
```

**Quick Evaluation (50 samples):**
```
[TO FILL]
Severity:        XX.XX%
Component:       XX.XX%
Reproducibility: XX.XX%
Overall:         XX.XX%
```

**Analysis:** [TO FILL]

---

## Comparative Analysis

### Performance Summary

| Configuration | Severity | Component | Reproducibility | Overall | Training Loss | Params |
|---------------|----------|-----------|-----------------|---------|---------------|--------|
| **V1 (baseline)** | 41.86% | 62.79% | 88.37% | 64.34% | 1.3216 | 16M |
| **V2-r4** | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | 8M |
| **V2 (r=8)** | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | 16M |
| **V2-r16** | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | 32M |

### Key Findings

**[TO FILL AFTER EXPERIMENTS]**

1. **Optimal rank:** r=[X] achieved best overall accuracy ([XX%])
2. **Overfitting behavior:** [Did r=16 show signs of overfitting? Compare train vs eval loss]
3. **Severity improvements:** r=[X] best for critical severity classification ([XX%] vs V1 41.86%)
4. **Diminishing returns:** [Does r=16 justify 2× parameters vs r=8?]

### Overfitting Analysis

**Training Loss vs Eval Loss:**
```
[TO FILL - Compare final training loss vs validation loss for each config]

V2-r4:  train=[X], eval=[Y], gap=[Y-X]
V2-r8:  train=[X], eval=[Y], gap=[Y-X]  
V2-r16: train=[X], eval=[Y], gap=[Y-X]

Interpretation: [Is gap increasing with rank? Sign of overfitting?]
```

### Recommendation

**Selected Configuration:** r=[TO FILL]

**Justification:** [TO FILL]
- Performance: [accuracy comparison]
- Efficiency: [params vs accuracy tradeoff]
- Robustness: [overfitting evidence]
- Production: [inference speed, memory requirements]

## Lessons Learned

### What Worked

**[TO FILL AFTER EXPERIMENTS]**

1. [Observation about rank vs performance]
2. [Unexpected findings]
3. [Confirmation or contradiction of hypothesis]

### What Didn't Work

**[TO FILL]**

1. [Any surprises or failures]
2. [Configurations that underperformed expectations]

### Future Improvements

**If More Time/Resources:**

1. **Full grid search:**
   - Ranks: [2, 4, 8, 12, 16, 24, 32]
   - Learning rates: [1e-4, 2e-4, 3e-4, 4e-4]
   - ~28 experiments × 2 hours = 56 hours

2. **Bayesian optimization:**
   - Use Optuna or Ray Tune
   - Automated search with early stopping
   - 10-15 experiments, intelligent sampling

3. **Ensemble approach:**
   - Train r=4, r=8, r=16 separately
   - Average predictions (soft voting)
   - Expected +1-3% accuracy boost

4. **Multi-objective optimization:**
   - Optimize for: accuracy + inference speed + memory
   - Pareto frontier analysis
   - Production deployment considerations

## Documentation for Technical Report

### Summary for Hyperparameter Optimization Section

**Strategy (3 points):** Systematic evaluation of LoRA rank (r=4, 8, 16) to balance expressiveness vs overfitting on 1399-example dataset. Held learning rate, batch size, epochs constant based on literature recommendations. Rationale: rank determines trainable parameters (8M→16M→32M), fundamental capacity tradeoff.

**Configurations (4 points):** Tested 3 configurations across 6 hours:
1. V2-r4: 8M params, quick eval (50 samples)
2. V2-r8: 16M params, full eval (300 samples) 
3. V2-r16: 32M params, quick eval (50 samples)

Results: [TABLE TO FILL]

**Comparison (3 points):** Analysis revealed [TO FILL: which rank optimal, why, overfitting evidence, severity accuracy gains]. Selected r=[X] for final model achieving [Y%] overall (+[Z]% vs V1 baseline). Training loss comparison showed [overfitting patterns/optimal convergence]. Recommendation based on accuracy-efficiency tradeoff and production constraints (13GB GPU memory).

---

**Status:** Experiment in progress  
**Last Updated:** February 9, 2026  
**Next Steps:** Complete V2 training → Run ablation experiments → Fill results → Document findings
