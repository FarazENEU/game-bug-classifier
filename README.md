# Game Bug Report Classifier & Triage System

> Fine-tuning a Large Language Model for intelligent bug report classification and developer triage

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Project Overview

This project fine-tunes a pre-trained Large Language Model to automatically classify and triage game bug reports. The system performs multiple tasks:

- **Severity Classification**: Critical, High, Medium, Low
- **Component Detection**: UI, Gameplay, Audio, Graphics, Network, etc.
- **Reproducibility Assessment**: Always, Sometimes, Rare
- **Developer Summary Generation**: Concise, actionable bug descriptions

### Real-World Impact
Game studios receive thousands of bug reports daily. This system can:
- Save developer time by automatically triaging reports
- Prioritize critical bugs for faster resolution
- Standardize bug report quality across platforms
- Surface patterns in recurring issues

## 🚀 Quick Start

### Environment Setup

**Requirements:**
- Python 3.8+
- CUDA-capable GPU (12GB+ VRAM recommended, or use Kaggle with 2× Tesla T4)
- 20GB disk space for model weights

**Installation:**

```bash
# Clone the repository
git clone <your-repo-url>
cd "LLM Fine Tuning"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Alternative: Install from requirements
pip install torch transformers peft bitsandbytes datasets accelerate tqdm
```

**Kaggle Setup (Recommended for Training):**
1. Create new Kaggle notebook
2. Enable GPU (Settings → Accelerator → GPU T4 x2)
3. Upload data splits as Kaggle dataset
4. Run: `!pip install transformers peft bitsandbytes accelerate`
5. Clone your repo or upload scripts directly

### Running the Demo

```bash
# Interactive classification demo
python scripts/demo.py --model_path final_model/ --mode interactive

# Pre-loaded examples
python scripts/demo.py --model_path final_model/ --mode examples

# Batch mode for video recording
python scripts/demo.py --model_path final_model/ --mode batch
```

### Training

```bash
# Train model with default hyperparameters (r=8, α=32)
python scripts/train.py \
    --train_path data/train_improved.jsonl \
    --val_path data/val.jsonl \
    --output_dir outputs/final_model \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4

# Train with custom LoRA rank (for hyperparameter optimization)
python scripts/train.py \
    --train_path data/train_improved.jsonl \
    --val_path data/val.jsonl \
    --output_dir outputs/model_r16 \
    --lora_r 16 \
    --lora_alpha 64
```

### Evaluation

```bash
# Evaluate fine-tuned model
python scripts/evaluate.py \
    --model_path final_model/ \
    --test_path data/test.jsonl \
    --num_samples 100

# Evaluate zero-shot baseline (no fine-tuning)
python scripts/evaluate_baseline.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --test_path data/test.jsonl \
    --num_samples 100
```

## 🏗️ Project Structure

```
.
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── setup.py                   # Package installation
├── .env.example               # Example environment variables
├── .gitignore                 # Git ignore patterns
│
├── configs/                   # Configuration files
│   ├── model_config.yaml      # Model hyperparameters
│   ├── training_config.yaml   # Training settings
│   └── data_config.yaml       # Data processing config
│
├── data/                      # Data directory (gitignored)
│   ├── raw/                   # Original bug reports
│   ├── processed/             # Cleaned and formatted data
│   ├── splits/                # Train/val/test splits
│   └── sample/                # Sample data for testing
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── data/                  # Data processing
│   │   ├── __init__.py
│   │   ├── collectors.py      # Data collection scripts
│   │   ├── preprocessors.py   # Data cleaning
│   │   └── formatters.py      # Format for fine-tuning
│   │
│   ├── models/                # Model code
│   │   ├── __init__.py
│   │   ├── model.py           # Model architecture
│   │   └── trainer.py         # Training logic
│   │
│   ├── evaluation/            # Evaluation code
│   │   ├── __init__.py
│   │   ├── metrics.py         # Custom metrics
│   │   └── evaluator.py       # Evaluation pipeline
│   │
│   └── utils/                 # Utilities
│       ├── __init__.py
│       ├── logger.py          # Logging setup
│       └── helpers.py         # Helper functions
│
├── scripts/                   # Executable scripts
│   ├── collect_data.py        # Data collection
│   ├── preprocess_data.py     # Data preprocessing
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   └── inference.py           # Inference/demo script
│
├── notebooks/                 # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_selection.ipynb
│   ├── 03_error_analysis.ipynb
│   └── 04_results_visualization.ipynb
│
├── tests/                     # Unit tests
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_model.py
│   └── test_evaluation.py
│
├── models/                    # Saved models (gitignored)
│   ├── base/                  # Pre-trained models
│   ├── checkpoints/           # Training checkpoints
│   └── final/                 # Final fine-tuned models
│
├── outputs/                   # Results and logs (gitignored)
│   ├── logs/                  # Training logs
│   ├── results/               # Evaluation results
│   └── predictions/           # Model predictions
│
└── docs/                      # Documentation
    ├── SETUP.md               # Setup instructions
    ├── METHODOLOGY.md         # Approach and methodology
    ├── RESULTS.md             # Results and analysis
    └── API.md                 # API documentation
