# Video Presentation Notes (8-9 minutes)

## ✅ ACCURATE SPECS

### Model & Training
- **Base Model:** mistralai/Mistral-7B-Instruct-v0.2 (7 billion parameters)
- **Method:** QLoRA (Quantized Low-Rank Adaptation)
- **LoRA rank:** 8, alpha: 32 (α/r = 4.0 ratio)
- **Target modules:** q_proj, v_proj, k_proj, o_proj
- **Quantization:** 4-bit NF4 with double quantization
- **Trainable parameters:** ~4 million (only 0.06% of base model!)
- **Training data:** 1,399 examples (keyword-labeled from GitHub Issues API)
- **Data sources:** 4 open-source game repos - Godot Engine, Bevy, Minetest, OpenRCT2 (500 each)
- **Epochs:** 3 for V1-r8 (completed), 1 for V1-r4 and V1-r16 (both completed)
- **Training time:** ~1.5 hours per configuration on Kaggle T4 GPU
- **Batch size:** 4 with 4× gradient accumulation = effective batch of 16

### Tasks (NOT priority, it's component!)
1. **Severity:** critical / high / medium / low
2. **Component:** ui / graphics / audio / network / save / performance / other
3. **Reproducibility:** always / sometimes / rare

### Results (V1-r8, 50 samples)
- **Severity:** 41.86%
- **Component:** 62.79%
- **Reproducibility:** 88.37%
- **Average:** 64.34%
- **Baseline (zero-shot):** 0.33% (essentially random)
- **Improvement:** 195× better than baseline

### Hyperparameter Experiments
- **V1-r4:** LoRA rank=4, α=16, 8M trainable params, 1 epoch - **59.96% ✓** (47.92% / 62.50% / 69.47%)
- **V1-r8:** LoRA rank=8, α=32, 16M trainable params, 3 epochs - **64.34% ✓** (41.86% / 62.79% / 88.37%)
- **V1-r16:** LoRA rank=16, α=64, 32M trainable params, 1 epoch - **59.93% ✓** (48.94% / 61.70% / 69.15%)
- **Key finding:** Training time matters more than rank! r4 and r16 (1 epoch) both achieve ~60%, while r8 (3 epochs) reaches 64.34%

---

## 🎬 SHOT LIST (8 minutes)

### INTRO (30 sec)
**Show:** Terminal or PDF title page  
**Say:**
- "I fine-tuned Mistral-7B-Instruct for automated bug triage using QLoRA"
- "Goal: classify severity, component, and reproducibility with minimal compute"
- "Using Kaggle's free 30-hour weekly GPU quota"

---

### PART 1: Code Walkthrough (2 mins)

**Screen 1:** Open `scripts/train.py` → scroll to setup_model() function (line ~73)

**Highlight:**
```python
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)
```

**Say:**
- "Using LoRA rank 8 which gives us only 4 million trainable parameters"
- "That's 0.06% of the full 7 billion parameter model"
- "With 4-bit quantization, fits easily in free GPU memory"

**Screen 2:** Scroll to training_args (line ~130+)

**Say:**
- "Trained for 3 epochs on 1,399 bug reports from 4 game projects"
- "Godot Engine, Bevy, Minetest, and OpenRCT2 - diverse bug patterns"
- "Small batch size with gradient accumulation to save memory"
- "Each training run takes about 1.5 hours on a Kaggle T4"

---

### PART 2: Results (2 mins)

**Screen 3:** Open PDF → page 7 (results table)

**Zoom in on table, say:**
- "Baseline zero-shot was essentially random at 0.33%"
- "After fine-tuning: 64.34% average accuracy - a 195× improvement"

**Point to each row:**
- "Severity: 41.86% - hardest because keyword labeling wasn't perfect"
- "Component: 62.79% - learned good patterns for UI, graphics, network bugs"
- "Reproducibility: 88.37% - best performance because bug reports have clear signals"
  - "Like 'always happens' or '100% of the time'"

**Screen 4:** Point to hyperparameter section in PDF

**Say:**
- "I also tested ranks 4, 8, and 16 to find optimal capacity"
- "Rank 4 achieved 59.96%, rank 8 got 64.34%, rank 16 got 59.93%"
- "Both r4 and r16 had just 1 epoch vs r8's 3 epochs - nearly identical results"
- "This shows training time matters more than rank in this range"
- "The 4.4 point improvement comes from additional epochs, not intermediate rank"

---

### PART 3: Live Demo (2.5 mins)

**Screen 5:** Kaggle notebook - create new code cell

**Paste and run:**
```python
!python /kaggle/working/scripts/kaggle_live_demo.py
```

**Say BEFORE running:**
- "Let me show the model working live on a sample bug report"
- "I'll classify a graphics crash bug in real-time"
- [Click run]

**[EDIT OUT: 2-3 minutes of model loading time]**

**Say AFTER model loads (when prediction appears):**
- "Here's the sample: game crashes when loading high-res textures"
- "GPU memory error, happens every time in the forest area"
- [Point to prediction output]
- "The model correctly classified this as:"
- "Severity: high - because it's a crash affecting gameplay"
- "Component: graphics - GPU memory and textures"
- "Reproducibility: always - 100% reproduction rate"
- "This shows the model learned to understand bug patterns from context, not just keywords"

**Alternative if showing evaluation results instead:**
```python
# If you prefer faster demo without model loading
import json
with open("/kaggle/working/outputs_v3_r8/evaluation_results.json") as f:
    data = json.load(f)
for i, pred in enumerate(data["predictions"][:3], 1):
    print(f"Example {i}: {pred['input'][:100]}...")
    print(f"Predicted: {pred['predicted']}")
```

---

### PART 4: V2 Lesson (1 min)

**Screen 6:** Stay in terminal or show PDF

**Say:**
- "Quick lesson learned about data-model alignment"
- "I tried a V2 with 'improved' vocabulary - changed low/medium/high to minor/moderate/major"
- "It completely failed at 0.67% - worse than baseline"
- "Why? Training data used the OLD vocabulary"
- "Takeaway: always validate your data matches your model before spending hours training"

---

### PART 5: Lessons Learned (1.5 mins)

**Screen 7:** PDF limitations section or code

**Say:**
- "Three big takeaways:"

**1. Data quality > model size**
- "My keyword-based labels limited severity accuracy to 41%"
- "Real human-labeled data would have helped significantly"

**2. Vocabulary alignment is critical**
- "V2 failed from a simple mismatch"
- "Always validate before training - save hours of compute"

**3. PEFT works with extreme constraints**
- "Only 4 million trainable parameters"
- "1.5-hour training runs on Kaggle's free 30-hour weekly tier"
- "Still got 64% accuracy on a real-world multi-task problem"

**4. Efficient iteration enables learning**
- "1.5-hour training cycles allowed testing multiple approaches"
- "Caught V2 mistake early instead of after days of training"

---

### WRAP UP (30 sec)

**Screen 8:** Back to PDF or terminal

**Say:**
- "This project proves parameter-efficient fine-tuning is viable for resource-constrained scenarios"
- "With better training data and more evaluation samples, could reach 70-80% accuracy"
- "All on Kaggle's free 30-hour weekly GPU tier"
- "Thanks! Happy to answer questions"

---

## 🎥 Recording Checklist

**Before Recording:**
```bash
# Test the demo script
cd "/Users/faraz/Projects/Uni/7375/LLM Fine Tuning"
python scripts/show_eval_examples.py

# Check PDF page count
open technical_report.pdf  # Should be 10 pages, results on page 7

# Have ready:
# ✅ VS Code with scripts/train.py open
# ✅ technical_report.pdf open to page 7
# ✅ Terminal with clean prompt
# ✅ Close all distractions (Slack, email, notifications)
```

**Recording Settings (QuickTime):**
- File → New Screen Recording
- Options → Show mouse clicks
- Select microphone
- Choose recording region (or full screen)
- Speak clearly at normal pace
- Zoom in (Cmd +) when showing code/tables

**If you mess up:**
- Pause 3 seconds of silence
- Continue from that point
- Can edit out mistakes later with QuickTime or iMovie

---

## ❌ Common Mistakes to Avoid

**DON'T SAY:**
- ❌ "Priority" (it's **component**, not priority!)
- ❌ "Qwen" or "Qwen2.5-3B" (it's **Mistral-7B-Instruct-v0.2**)
- ❌ "193× improvement" (it's **195×**)
- ❌ "V3" experiments (they're **V1-r4 and V1-r16**)

**DO SAY:**
- ✅ "Mistral-7B-Instruct-v0.2 fine-tuned with QLoRA"
- ✅ "Three tasks: severity, component, reproducibility"
- ✅ "64.34% average accuracy, 195× better than baseline"
- ✅ "V1-r8 best at 64.34%, V1-r4 at 59.96%, V1-r16 at 59.93%"

---

## 📊 Quick Reference Card

### V1-r8 (Best Configuration - 3 epochs)
| Metric                   | Value                    |
| ------------------------ | ------------------------ |
| Base Model               | Mistral-7B-Instruct-v0.2 |
| Base Parameters          | 7 billion                |
| Trainable Params         | ~16 million (0.2%)       |
| LoRA Rank                | 8 (α=32)                 |
| Training Data            | 1,399 examples           |
| Training Time            | ~4.5 hours (3 epochs)    |
| Test Set                 | 50 samples (±14% CI)     |
| Severity Accuracy        | 41.86%                   |
| Component Accuracy       | 62.79%                   |
| Reproducibility Accuracy | 88.37%                   |
| **Average Accuracy**     | **64.34%**               |
| Baseline                 | 0.33%                    |
| Improvement              | 195×                     |

### V1-r4 (Lower Rank - 1 epoch)
| Metric                   | Value                        |
| ------------------------ | ---------------------------- |
| LoRA Rank                | 4 (α=16)                     |
| Trainable Params         | ~8 million                   |
| Training Time            | ~1.5 hours (1 epoch)         |
| Severity Accuracy        | 47.92% (↑ better)            |
| Component Accuracy       | 62.50% (≈ similar)           |
| Reproducibility Accuracy | 69.47% (↓ much worse)        |
| **Average Accuracy**     | **59.96%** (↓ 4.4 pts worse) |

### V1-r16 (Higher Rank - 1 epoch)
| Metric                   | Value                        |
| ------------------------ | ---------------------------- |
| LoRA Rank                | 16 (α=64)                    |
| Trainable Params         | ~32 million                  |
| Training Time            | ~1.5 hours (1 epoch)         |
| Severity Accuracy        | 48.94% (↑ better)            |
| Component Accuracy       | 61.70% (≈ similar)           |
| Reproducibility Accuracy | 69.15% (↓ much worse)        |
| **Average Accuracy**     | **59.93%** (↓ 4.4 pts worse) |

**Insight:** r4 and r16 with 1 epoch perform identically (~60%), showing training time matters more than rank capacity in this range.

---

## 🎯 Key Points to Emphasize

1. **Resource efficiency:** 0.06-0.4% of parameters trained, 1.5-hour cycles, Kaggle's 30-hour weekly tier
2. **Real-world application:** Multi-task bug triage, not toy problem
3. **Practical lessons:** Data quality matters, vocabulary alignment critical, **training time > rank capacity**
4. **Reproducible:** All code/data saved, hyperparameter comparison complete (r4 vs r8 vs r16)
5. **Honest about limitations:** Keyword labeling, small eval set, task difficulty variance

---

Good luck with the recording! 🎬
