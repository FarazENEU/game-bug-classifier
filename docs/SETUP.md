# Setup Instructions

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended: 16GB+ VRAM)
- 50GB+ free disk space

## Installation Steps

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd "LLM Fine Tuning"
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

For development (includes testing and code quality tools):
```bash
pip install -e ".[dev]"
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and add your tokens:
- **HF_TOKEN**: Get from [Hugging Face Settings](https://huggingface.co/settings/tokens)
- **GITHUB_TOKEN**: Get from [GitHub Settings](https://github.com/settings/tokens)
- **WANDB_API_KEY**: (Optional) Get from [Weights & Biases](https://wandb.ai/authorize)

### 5. Create Required Directories

```bash
mkdir -p data/{raw,processed,splits,sample}
mkdir -p models/{base,checkpoints,final}
mkdir -p outputs/{logs,results,predictions}
mkdir -p cache
```

### 6. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

## Quick Start

### Collect Data
```bash
python scripts/collect_data.py
```

### Preprocess Data
```bash
python scripts/preprocess_data.py
```

### Train Model
```bash
python scripts/train.py --config configs/training_config.yaml
```

### Evaluate Model
```bash
python scripts/evaluate.py --model_path models/final/best_model
```

### Run Inference
```bash
python scripts/inference.py --interactive
```

## Troubleshooting

### Out of Memory (OOM) Errors
- Reduce `per_device_train_batch_size` in `configs/training_config.yaml`
- Enable gradient checkpointing
- Use 4-bit quantization

### Slow Training
- Increase `gradient_accumulation_steps`
- Enable mixed precision training (bf16/fp16)
- Use fewer dataloader workers

### Data Collection Issues
- Ensure GitHub token has proper permissions
- Check rate limits (GitHub API: 5000 requests/hour)
- Try collecting from different repositories

## Hardware Requirements

### Minimum
- GPU: 8GB VRAM (using 4-bit quantization)
- RAM: 16GB
- Storage: 30GB

### Recommended
- GPU: 24GB VRAM (RTX 3090/4090, A5000)
- RAM: 32GB+
- Storage: 100GB SSD

### Cloud Options
- Google Colab Pro+ (A100)
- AWS EC2 g5.xlarge
- Paperspace Gradient
- Lambda Labs
