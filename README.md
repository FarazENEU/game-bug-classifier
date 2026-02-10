# Game Bug Report Classification with QLoRA

> Parameter-efficient fine-tuning of Mistral-7B-Instruct for automated bug triage

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

This project fine-tunes Mistral-7B-Instruct using QLoRA (Quantized Low-Rank Adaptation) to classify game bug reports into three categories:

- **Severity**: critical / high / medium / low
- **Component**: ui / graphics / audio / network / save / performance / other
- **Reproducibility**: always / sometimes / rare

**Key Results:**
- **64.34%** average accuracy on multi-task classification (V1-r8, 3 epochs)
- **195× improvement** over zero-shot baseline (0.33%)
- Trained on **1,399 examples** from 4 game repositories (Godot, Bevy, Minetest, OpenRCT2)
- **~4.5 hours** total training time on Kaggle's free T4 GPU

## 🎓 Academic Context

This is a course project demonstrating:
- Parameter-efficient fine-tuning (PEFT) with limited compute resources
- Multi-task classification with structured outputs
- Bootstrapped labeling for specialized domains
- Hyperparameter experiments (LoRA rank r ∈ {4, 8, 16})

**Technical Report:** [`technical_report.pdf`](technical_report.pdf)

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **GPU**: 15GB+ VRAM (Tesla T4 or better) or use [Kaggle](https://www.kaggle.com/) with free GPU
- **Disk**: ~10GB for model weights and data

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd "LLM Fine Tuning"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-local.txt  # For local training
# OR
pip install -r requirements-kaggle.txt  # For Kaggle environment
```

### Run Demo

View sample predictions from the trained model:

```bash
python scripts/show_eval_examples.py
```

This displays:
- 3 example bug reports with predictions
- Overall metrics (64.34% average accuracy)
- Per-task breakdown (severity, component, reproducibility)

## 🔬 Training

### Training on Kaggle (Recommended)

**Why Kaggle?** Free T4 GPU, 30-hour weekly quota, pre-installed libraries.

1. **Setup Kaggle notebook:**
   - Create new notebook at [kaggle.com/code](https://www.kaggle.com/code)
   - Enable GPU: Settings → Accelerator → GPU T4
   - Clone repository or upload scripts

2. **Install dependencies:**
   ```python
   !pip install -q transformers peft bitsandbytes accelerate datasets
   ```

3. **Run training:**
   ```bash
   cd /kaggle/working/game-bug-classifier
   
   python scripts/train.py \
       --train_path data/train.jsonl \
       --val_path data/val.jsonl \
       --output_dir /kaggle/working/outputs_v1_r8 \
       --num_epochs 3 \
       --batch_size 4 \
       --lora_r 8 \
       --lora_alpha 32
   ```

4. **Evaluate:**
   ```bash
   python scripts/evaluate.py \
       --model_path /kaggle/working/outputs_v1_r8/final_model \
       --test_path data/test.jsonl \
       --output_file evaluation_results.json
   ```

### Training Locally (If you have GPU)

```bash
python scripts/train.py \
    --train_path data/train.jsonl \
    --val_path data/val.jsonl \
    --output_dir outputs/model \
    --num_epochs 3 \
    --batch_size 4
```

**Note:** Training requires ~13GB VRAM with 4-bit quantization. Reduce `--batch_size 2` if OOM errors occur.

### Hyperparameter Experiments

Test different LoRA ranks:

```bash
# Rank 4 (8M trainable params)
python scripts/train.py --lora_r 4 --lora_alpha 16 --output_dir outputs_r4

# Rank 8 (16M trainable params) - BEST
python scripts/train.py --lora_r 8 --lora_alpha 32 --output_dir outputs_r8

# Rank 16 (32M trainable params)
python scripts/train.py --lora_r 16 --lora_alpha 64 --output_dir outputs_r16
```

## 📂 Project Structure

```
.
├── README.md                     # This file
├── technical_report.pdf          # Academic paper
├── technical_report.tex          # LaTeX source
├── requirements-kaggle.txt       # Kaggle dependencies
├── requirements-local.txt        # Local dependencies
├── LICENSE
│
├── data/                         # Training data (JSONL format)
│   ├── train.jsonl               # 1,399 training examples
│   ├── val.jsonl                 # 299 validation examples
│   └── test.jsonl                # 300 test examples
│
├── scripts/                      # Executable scripts
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   ├── evaluate_baseline.py      # Zero-shot baseline
│   ├── show_eval_examples.py     # Display sample predictions
│   ├── kaggle_live_demo.py       # Live inference demo
│   ├── collect_data.py           # Data collection from GitHub
│   ├── preprocess_data.py        # Data preprocessing
│   ├── improve_labels.py         # Label improvement experiments
│   ├── quick_eval.py             # Quick 50-sample evaluation
│   └── ...
│
├── src/                          # Source code modules
│   ├── data/                     # Data processing
│   ├── models/                   # Model definitions
│   ├── evaluation/               # Evaluation metrics
│   └── utils/                    # Utilities
│
├── final_model/                  # Trained models
│   ├── outputs/final_model/      # V1-r8 adapter weights
│   └── evaluation_results.json   # Evaluation metrics
│
├── baseline_zero_shot_results.json  # Baseline metrics
└── configs/                      # Configuration files
```

## 📊 Results Summary

| Configuration        | Severity | Component | Reproducibility | **Overall**  |
| -------------------- | -------- | --------- | --------------- | ------------ |
| Baseline (zero-shot) | 1.00%    | 0.00%     | 0.00%           | **0.33%**    |
| V1-r4 (1 epoch)      | 47.92%   | 62.50%    | 69.47%          | **59.96%**   |
| V1-r8 (3 epochs)     | 41.86%   | 62.79%    | 88.37%          | **64.34%** ✓ |
| V1-r16 (1 epoch)     | 48.94%   | 61.70%    | 69.15%          | **59.93%**   |

**Key findings:**
- r=8 with 3 epochs achieved best performance
- Training time matters more than rank capacity (r4 ≈ r16 with 1 epoch each)
- Reproducibility easiest to classify (88.37%), severity hardest (41.86%)

## 🔧 Configuration

Training hyperparameters (from `scripts/train.py`):

```python
# QLoRA Configuration
quantization: 4-bit NF4
lora_r: 8             # LoRA rank
lora_alpha: 32        # Scaling factor (α/r = 4.0)
target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

# Training
batch_size: 4                      # Per-device
gradient_accumulation_steps: 4     # Effective batch = 16
learning_rate: 2e-4
num_epochs: 3
max_length: 512 tokens
optimizer: AdamW with cosine annealing
warmup: 10%
```

## 🧪 Evaluation

Evaluate a trained model:

```bash
python scripts/evaluate.py \
    --model_path final_model/outputs/final_model \
    --test_path data/test.jsonl \
    --output_file results.json
```

Quick evaluation (50 samples for faster iteration):

```bash
python scripts/quick_eval.py \
    --model_path final_model/outputs/final_model \
    --test_path data/test.jsonl
```

Baseline comparison:

```bash
python scripts/evaluate_baseline.py \
    --test_path data/test.jsonl \
    --num_samples 50
```

## 📝 Data Collection

The training data was collected from GitHub Issues API:

```bash
python scripts/collect_data.py \
    --repos godotengine/godot bevyengine/bevy minetest/minetest OpenRCT2/OpenRCT2 \
    --output_dir data/raw \
    --max_per_repo 500
```

Then preprocessed and split:

```bash
python scripts/preprocess_data.py \
    --input_dir data/raw \
    --output_dir data \
    --train_split 0.7 \
    --val_split 0.15 \
    --test_split 0.15
```

## 📖 Citation

If you use this code or methodology for your work, please cite:

```bibtex
@misc{bug_classifier_2026,
  title={Game Bug Report Classification Using Fine-Tuned LLMs with Bootstrapped Labels},
  author={[Your Name]},
  year={2026},
  howpublished={University Assignment - COMP 7375},
  note={QLoRA fine-tuning of Mistral-7B-Instruct}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Base Model**: Mistral-7B-Instruct-v0.2 by Mistral AI
- **Training Platform**: Kaggle (free GPU tier)
- **Data Sources**: GitHub Issues from Godot, Bevy, Minetest, OpenRCT2
- **Libraries**: Hugging Face Transformers, PEFT, BitsAndBytes

## 📧 Contact

For questions about this project, please contact [your-email] or open an issue in the repository.

---

**Project Status**: ✅ Complete (February 2026)
