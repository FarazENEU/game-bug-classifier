# Methodology

## 1. Dataset Preparation

### 1.1 Data Collection Strategy

We collect bug reports from multiple open-source game repositories on GitHub:

**Selected Repositories:**
- **Godot Engine**: AAA-quality game engine with extensive bug reporting
- **Bevy**: Modern Rust game engine with detailed technical issues
- **Minetest**: Open-source voxel game with community-driven bug reports
- **OpenRCT2**: Reverse-engineered game with well-documented issues

**Collection Criteria:**
- Issues labeled as "bug", "defect", "crash", or "error"
- Both open and closed issues (learning from resolved bugs)
- Minimum 50 characters of description
- Issues with sufficient context for classification

**Tools Used:**
- GitHub API via PyGithub
- Rate limiting: 5000 requests/hour (authenticated)
- Target: 5000-10000 bug reports total

### 1.2 Data Preprocessing Pipeline

**Stage 1: Cleaning**
- Remove duplicate reports (by title + description hash)
- Filter out incomplete reports (missing title or body)
- Remove non-English reports (using langdetect)
- Normalize whitespace and formatting

**Stage 2: Labeling**
We extract/infer three types of labels:

1. **Severity** (Critical/High/Medium/Low)
   - Keywords in title/body
   - Issue priority labels
   - Presence of words like "crash", "data loss", "security"

2. **Component** (UI/Gameplay/Audio/Graphics/Network/Performance/Save/Other)
   - Content analysis of bug description
   - File paths mentioned in reports
   - Keywords and technical terms

3. **Reproducibility** (Always/Sometimes/Rare)
   - Steps to reproduce section
   - Comments about consistency
   - Keywords like "always", "intermittent", "random"

**Stage 3: Formatting**
Convert to instruction-tuning format:
```json
{
  "instruction": "Classify this bug report and provide a developer summary.",
  "input": "[BUG REPORT TEXT]",
  "output": "Severity: High\nComponent: Graphics\nReproducibility: Always\nSummary: [GENERATED SUMMARY]"
}
```

### 1.3 Data Splits

- **Training**: 70% (stratified by severity)
- **Validation**: 15% (for hyperparameter tuning)
- **Test**: 15% (held out for final evaluation)

Stratification ensures balanced representation of all severity levels.

## 2. Model Selection

### 2.1 Base Model Choice

**Selected Model**: Mistral-7B-Instruct-v0.2

**Justification:**
- **Instruction-following**: Pre-trained on instruction datasets
- **Size**: 7B parameters - balance between capability and resource requirements
- **Performance**: State-of-the-art on various benchmarks
- **Licensing**: Apache 2.0 - commercially friendly
- **Context window**: 8K tokens - sufficient for long bug reports

**Alternative Models Considered:**
- GPT-2 (too small, 1.5B params)
- Llama-2-7B (comparable, but Mistral performs better on instruction tasks)
- Larger models (13B+) - resource constraints

### 2.2 Fine-Tuning Approach

**Parameter-Efficient Fine-Tuning (PEFT) with LoRA:**
- **Trainable parameters**: <1% of total model
- **LoRA rank (r)**: 8
- **Alpha**: 32
- **Target modules**: Query, Key, Value, Output projections
- **Memory benefits**: Can train on consumer GPUs

**Quantization:**
- **4-bit quantization** using bitsandbytes
- Reduces memory footprint by ~75%
- Minimal impact on model quality

## 3. Training Setup

### 3.1 Training Configuration

```yaml
Learning Rate: 2e-5
Batch Size: 4 (per device)
Gradient Accumulation: 4 steps
Effective Batch Size: 16
Epochs: 3
Optimizer: AdamW
LR Schedule: Cosine with warmup
Warmup Steps: 500
Weight Decay: 0.01
Max Sequence Length: 512 tokens
```

### 3.2 Training Strategy

1. **Checkpointing**: Save every 500 steps
2. **Evaluation**: Evaluate on validation set every 500 steps
3. **Early Stopping**: Monitor eval_loss, save best model
4. **Gradient Checkpointing**: Reduce memory usage
5. **Mixed Precision**: BF16 training for A100/H100

### 3.3 Monitoring

- **Weights & Biases**: Track loss, metrics, system stats
- **TensorBoard**: Visualize training curves
- **Custom metrics**: Severity accuracy, component F1-score

## 4. Hyperparameter Optimization

### 4.1 Search Strategy

We test **3 hyperparameter configurations**:

**Configuration 1: Conservative**
- Learning Rate: 1e-5
- LoRA Rank: 4
- Batch Size: 8
- Focus: Stability and baseline performance

**Configuration 2: Standard (Recommended)**
- Learning Rate: 2e-5
- LoRA Rank: 8
- Batch Size: 4
- Focus: Balance between quality and training time

**Configuration 3: Aggressive**
- Learning Rate: 5e-5
- LoRA Rank: 16
- Batch Size: 2
- Focus: Maximum capacity, risk of overfitting

### 4.2 Evaluation Criteria

For each configuration, we measure:
- Training loss convergence
- Validation loss
- Severity classification accuracy
- Component classification F1-score
- Summary quality (ROUGE scores)
- Training time and memory usage

## 5. Evaluation Methodology

### 5.1 Metrics

**Classification Metrics:**
- **Accuracy**: Overall and per-class
- **Precision/Recall/F1**: For each category
- **Confusion Matrix**: Identify misclassification patterns

**Generation Metrics:**
- **ROUGE-L**: Summary overlap with human-written summaries
- **BLEU**: N-gram overlap
- **Perplexity**: Model confidence

**Baseline Comparison:**
- Compare against zero-shot Mistral-7B (no fine-tuning)
- Compare against rule-based classifier (keyword matching)

### 5.2 Error Analysis

We perform systematic error analysis:

1. **False Positives/Negatives**: Where does classifier fail?
2. **Confusion Patterns**: Which categories are confused?
3. **Input Characteristics**: What makes reports hard to classify?
4. **Summary Quality**: Manual review of generated summaries

### 5.3 Qualitative Evaluation

- Manual review of 100 random test predictions
- Assessment of summary usefulness for developers
- Identification of model biases or limitations

## 6. Inference Pipeline

### 6.1 Model Serving

- Load fine-tuned LoRA weights
- Efficient batching for multiple reports
- Streaming output for long summaries

### 6.2 Interface

- Command-line interface for batch processing
- Interactive mode for single reports
- API endpoint (optional) for integration

## 7. Reproducibility

All experiments are fully reproducible:
- Fixed random seeds (42)
- Versioned dependencies (requirements.txt)
- Logged hyperparameters
- Saved model checkpoints
- Documented data sources

## 8. Ethical Considerations

- **Bias**: Bug reports may reflect biases in reporting patterns
- **Privacy**: No personal information included in training data
- **Transparency**: Clear documentation of limitations
- **Intended Use**: Tool to assist developers, not replace human judgment
