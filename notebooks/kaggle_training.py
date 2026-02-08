# Kaggle Notebook - Game Bug Classifier Training
# Copy-paste these cells into your Kaggle notebook

# ============================================================
# CELL 1: Setup
# ============================================================
# Clone repository
!git clone https://github.com/YOUR_USERNAME/game-bug-classifier.git
%cd game-bug-classifier

# Check GPU
!nvidia-smi

# ============================================================
# CELL 2: Install Dependencies
# ============================================================
!pip install -q transformers>=4.35.0 datasets>=2.14.0 peft>=0.6.0 bitsandbytes>=0.41.0 accelerate>=0.24.0

# ============================================================
# CELL 3: Verify Data
# ============================================================
import os

# Check if data is uploaded
data_path = "/kaggle/input/bug-data/"  # Adjust if you named it differently
files = os.listdir(data_path) if os.path.exists(data_path) else []
print(f"Found files: {files}")

# Should see: train.jsonl, val.jsonl, test.jsonl

# ============================================================
# CELL 4: Start Training (Choose ONE option)
# ============================================================

# OPTION A: Full Mistral-7B (3-4 hours, best quality)
!python scripts/train.py \
    --model_name "mistralai/Mistral-7B-Instruct-v0.2" \
    --train_path "/kaggle/input/bug-data/train.jsonl" \
    --val_path "/kaggle/input/bug-data/val.jsonl" \
    --output_dir "/kaggle/working/outputs" \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4

# OPTION B: GPT-2 (1-2 hours, faster, still good)
# Uncomment if you want faster training:
# !python scripts/train.py \
#     --model_name "gpt2" \
#     --train_path "/kaggle/input/bug-data/train.jsonl" \
#     --val_path "/kaggle/input/bug-data/val.jsonl" \
#     --output_dir "/kaggle/working/outputs" \
#     --num_epochs 5 \
#     --batch_size 8 \
#     --learning_rate 5e-5

# ============================================================
# CELL 5: Package Model for Download
# ============================================================
# Run this after training completes
!zip -r bug_classifier_model.zip /kaggle/working/outputs/models/final_model/

# Right-click on bug_classifier_model.zip in file browser and download

# ============================================================  
# CELL 6: Quick Test (Optional)
# ============================================================
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load model
base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto")
model = PeftModel.from_pretrained(base_model, "/kaggle/working/outputs/models/final_model")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

# Test inference
test_input = """### Instruction:
Classify this bug report and provide a structured analysis.

### Input:
Title: Game crashes on startup

Description: When I try to launch the game, it immediately crashes with no error message. Happens every time.

### Output:
"""

inputs = tokenizer(test_input, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=256, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
