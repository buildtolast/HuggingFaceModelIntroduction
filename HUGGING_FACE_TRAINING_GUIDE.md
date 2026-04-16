# Comprehensive Guide to Training Hugging Face Models

## Table of Contents
1. [Training Methods Overview](#training-methods-overview)
2. [Beginner Level](#beginner-level)
3. [Intermediate Level](#intermediate-level)
4. [Advanced Level](#advanced-level)
5. [Training Formats & Data Preparation](#training-formats--data-preparation)
6. [Training Infrastructure](#training-infrastructure)
7. [Common Training Frameworks](#common-training-frameworks)
8. [Datasets](#datasets)

---

## Training Methods Overview

### Full Fine-Tuning
**When to Use:** 
- You have substantial compute resources (high-end GPUs, multiple GPUs, or TPUs)
- You need maximum model quality and performance
- You want to update all model parameters
- Your use case requires complete model adaptation

**Requirements:**
- High VRAM: 24GB+ for 7B models, 40GB+ for 13B models
- Time: Hours to days depending on dataset size
- Storage: Full model weights (7B = ~14GB in fp16)

**Process:**
1. Load pre-trained model
2. Add task-specific head (if needed)
3. Train on entire dataset with all parameters updated
4. Save complete new model weights

**Advantages:**
- Maximum flexibility and model quality
- Can change model behavior comprehensively
- No adapter complexity

**Disadvantages:**
- Highest compute cost
- Risk of catastrophic forgetting
- Slow training compared to parameter-efficient methods
- Large artifact sizes

---

### LoRA (Low-Rank Adaptation)
**When to Use:**
- You have moderate GPU VRAM (16-24GB)
- You want faster training than full fine-tuning
- You need smaller checkpoints for deployment
- You want to maintain model stability

**How It Works:**
Instead of updating all weights W, LoRA adds trainable low-rank matrices A and B:
```
W_new = W + A × B
```
- A: (d × r) matrix
- B: (r × d) matrix  
- r: rank (typically 8-64, much smaller than d)

Only A and B are trained; base model weights remain frozen.

**Requirements:**
- VRAM: 16-24GB for 7B models
- Training time: 2-4x faster than full fine-tuning
- Checkpoint size: 1-5% of full model

**Advantages:**
- 10-100x fewer parameters to train
- Much smaller checkpoints (10-500MB vs 14GB)
- Faster training
- Can stack multiple adapters
- Better stability than full fine-tuning

**Disadvantages:**
- Slightly lower quality than full fine-tuning
- Inference needs base model + adapter
- Limited rank can bottleneck learning

**Code Example:**
```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

lora_config = LoraConfig(
    r=16,                          # Rank of low-rank matrices
    lora_alpha=32,                 # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Apply to attention
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(model, lora_config)

# Train normally with Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
```

---

### QLoRA (Quantized LoRA)
**When to Use:**
- You have limited GPU VRAM (8-16GB, or even T4/L4 from Colab)
- You want to fine-tune large models on consumer hardware
- Cost is a major constraint
- You're willing to accept minor quality trade-offs

**How It Works:**
1. Load base model in 4-bit quantization (NF4 format)
2. Train LoRA adapters in 16-bit or bfloat16
3. Quantization error is minimized, not amplified

**Benefits Over LoRA:**
- 50-70% less VRAM than LoRA
- Run 7B, 13B, even 33B models on single consumer GPU
- Maintain near-LoRA quality
- Enable fine-tuning on Google Colab T4/L4

**Requirements:**
- VRAM: 8-12GB for 7B models, 16GB for 13B models
- Install bitsandbytes: `pip install bitsandbytes`
- Training time: Similar to LoRA

**Requirements:**
- VRAM: 8-12GB
- Slightly slower training than LoRA (quantization overhead)
- Checkpoint size: Same as LoRA

**Advantages:**
- Enables fine-tuning large models on consumer hardware
- Massive VRAM reduction
- Competitive with full fine-tuning quality
- Most accessible method for hobbyists

**Disadvantages:**
- Slight training speed overhead from quantization
- Cannot merge adapters easily (quantization is lossy)
- Inference still requires base model + adapter

**Code Example:**
```python
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments
)

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NF4 quantization (better than default)
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,      # Double quantization
)

# Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    quantization_config=bnb_config,
    device_map="auto",
)

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()

# Save adapter only
model.save_pretrained("my-qlora-adapter")
```

---

### DPO (Direct Preference Optimization)
**When to Use:**
- You have preference data (chosen vs rejected responses)
- You want alignment without RLHF complexity
- You want to replace RLHF with simpler approach
- Your task involves subjective quality judgments

**How It Works:**
DPO directly optimizes language models on preference data without:
- Separate reward model training
- Complex reinforcement learning (PPO)
- Multi-stage pipeline

Instead of: SFT → Reward Model → PPO → Aligned Model
DPO does: SFT → DPO Loss → Aligned Model

**Math:**
```
Loss = -log(sigmoid(β * (log π(chosen) - log π(reference) - log π_ref(chosen) + log π_ref(rejected))))
```

Where:
- β: Temperature parameter (0.1-0.5 typical)
- π: Current policy model
- π_ref: Reference model (usually the SFT base)

**Requirements:**
- Preference pairs dataset (chosen, rejected responses)
- Less compute than RLHF (no need for reward model)
- Training time: Similar to SFT

**Advantages:**
- Much simpler than RLHF
- Single training phase instead of three
- No reward model engineering
- Stable training (no PPO hyperparameter tuning)
- Produces models comparable to RLHF quality

**Disadvantages:**
- Requires preference pairs (not just good examples)
- Can overfit to training preferences
- Sensitive to data quality
- Beta parameter needs tuning

**Code Example:**
```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Load SFT model first
model = AutoModelForCausalLM.from_pretrained("path/to/sft-model")
ref_model = AutoModelForCausalLM.from_pretrained("path/to/sft-model")
tokenizer = AutoTokenizer.from_pretrained("path/to/sft-model")

# Load preference dataset
train_dataset = load_dataset("your-preference-dataset")

# Configure DPO
dpo_config = DPOConfig(
    output_dir="dpo-model",
    learning_rate=5e-7,  # Lower than SFT
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    beta=0.1,  # KL divergence penalty
    max_prompt_length=512,
    max_length=1024,
)

# Train
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

---

### RLHF (Reinforcement Learning from Human Feedback)
**When to Use:**
- You have access to human raters or preference data at scale
- You need production-grade alignment (like ChatGPT)
- You have substantial compute for three-stage training
- You want maximum control over model behavior

**Three-Stage Pipeline:**

**Stage 1: Supervised Fine-Tuning (SFT)**
- Fine-tune on high-quality demonstration data
- Create foundation model with basic instruction following
- Data: (prompt, ideal_response) pairs

**Stage 2: Reward Model Training**
- Train separate reward model on preference data
- Input: (prompt, response_1, response_2)
- Output: scalar reward (which response is better)
- This model learns human preferences

**Stage 3: Policy Optimization with PPO**
- Use reward model to guide training
- Apply Proximal Policy Optimization
- Optimize base model to maximize reward
- Include KL divergence penalty to prevent divergence

**Requirements:**
- Compute: Very high (3 separate training runs)
- Time: Days/weeks for large models
- Expertise: Complex hyperparameter tuning
- Data: Preference annotations at scale

**Advantages:**
- Most flexible alignment method
- Can optimize for complex, multi-faceted preferences
- Proven to work (used by OpenAI, DeepSeek, others)
- Can continuously improve with new data

**Disadvantages:**
- Highest complexity
- Requires 3 separate models in memory
- PPO is unstable (sensitive to hyperparameters)
- Expensive (reward model training, PPO rollouts)
- Long feedback loop

**High-Level Code:**
```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoModelForCausalLM, AutoTokenizer

# Stage 1: SFT (standard Trainer)
sft_model = train_sft(base_model, sft_dataset)

# Stage 2: Reward Model
reward_model = train_reward_model(sft_model, preference_dataset)

# Stage 3: PPO
ppo_config = PPOConfig(
    learning_rate=1.41e-5,
    log_with="wandb",
)

model = AutoModelForCausalLMWithValueHead.from_pretrained(sft_model)
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=sft_model,
    tokenizer=tokenizer,
    dataset=prompt_dataset,
    data_collator=collate_fn,
)

# Training loop
for epoch in range(num_epochs):
    for batch in prompt_dataset:
        # Generate responses
        query_tensors = batch["input_ids"]
        response_tensors = ppo_trainer.generate(query_tensors)
        
        # Get rewards from reward model
        rewards = reward_model(response_tensors)
        
        # PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
```

---

### Instruction Tuning vs General Fine-Tuning

**General Fine-Tuning:**
- Train on task-specific data
- Focus: Improve on specific downstream task
- Example: Train on customer support tickets
- Data format: (text, label) or (prompt, completion)
- Best for: Specialized domains, classification, generation

**Instruction Tuning:**
- Train on instruction-following data
- Focus: Make model follow natural language instructions
- Example: "Summarize this: [text]" → summary
- Data format: Structured instructions with examples
- Best for: Chat models, assistants, multi-task learning
- Creates more generalizable, instruction-aware models

**Data Format Comparison:**

General Fine-Tuning:
```json
{"text": "This movie was great", "label": "positive"}
```

Instruction Tuning:
```json
{
  "instruction": "Classify the sentiment",
  "input": "This movie was great",
  "output": "positive"
}
```

---

### Multi-Task Learning
**When to Use:**
- You have multiple related tasks
- You want one model to handle many domains
- You want better generalization

**Approach:**
Mix datasets from different tasks in training. Model learns to handle all tasks simultaneously.

```python
# Create combined dataset with task identifiers
mixed_dataset = concatenate_datasets([
    task1_dataset.map(lambda x: {**x, "task": "task1"}),
    task2_dataset.map(lambda x: {**x, "task": "task2"}),
    task3_dataset.map(lambda x: {**x, "task": "task3"}),
])

# Train on mixed data
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=mixed_dataset,
)
```

---

## Beginner Level

### Simple Fine-Tuning with Hugging Face Trainer

**Simplest Example - 20 Lines of Code:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# 1. Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 2. Load dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1000]")

# 3. Tokenize
def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

# 4. Set up training
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10,
    save_total_limit=2,
)

# 5. Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
)
trainer.train()

# 6. Save
trainer.save_model("my-fine-tuned-model")
```

**Key Concepts:**
- `Trainer`: Handles training loop automatically
- `TrainingArguments`: Configure batch size, epochs, learning rate
- `tokenize()`: Prepare text for model input
- Model automatically saves checkpoints

---

### Google Colab Setup (Free GPU)

```python
# In Colab cell 1: Install dependencies
!pip install transformers datasets torch accelerate

# In Colab cell 2: GPU Check
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# In Colab cell 3: Load small model (works on free T4)
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")  # Small model
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# In Colab cell 4: Train (as in example above)
```

**Colab Tips:**
- Free tier has T4 GPU (16GB VRAM)
- Enough for small models and LoRA
- Session times out after 12 hours
- Save to Google Drive: `!cp -r ./results /content/drive/MyDrive/`

---

### Common Pitfalls

1. **Forgetting to set tokenizer padding token:**
```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

2. **Not shuffling dataset:**
```python
dataset = dataset.shuffle(seed=42)
```

3. **Too large batch size → OOM:**
- Start small: batch_size=4 or 8
- Increase if you have VRAM

4. **Learning rate too high → training diverges:**
- Default 5e-5 is usually safe
- For fine-tuning: 1e-5 to 5e-4 range

5. **Tokenizing without truncation → slow/OOM:**
```python
# Good
return tokenizer(text, truncation=True, max_length=512)

# Bad
return tokenizer(text)  # Could be very long
```

---

## Intermediate Level

### LoRA with PEFT Library

**Full LoRA Training Example:**

```python
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")

# Configure LoRA
lora_config = LoraConfig(
    r=8,                    # Low rank
    lora_alpha=16,          # Scaling
    target_modules=["q_proj", "v_proj"],  # Attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
print(model.print_trainable_parameters())  # See how many params trainable

# Prepare data
dataset = load_dataset("your-dataset")

def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

# Training args
training_args = TrainingArguments(
    output_dir="./lora-model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Simulate batch_size 16
    learning_rate=3e-4,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
)
trainer.train()

# Save only LoRA weights (small!)
model.save_pretrained("./lora-weights")

# Load later for inference
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
model = PeftModel.from_pretrained(base_model, "./lora-weights")
```

**Output:**
```
trainable params: 4,194,304 || all params: 6,738,415,616 || trainable%: 0.06
```
Only 0.06% of parameters trained!

---

### Multi-GPU Training

**Using Distributed Data Parallel (DDP):**

```bash
# Run with multiple GPUs
torchrun --nproc_per_node=4 train.py
```

**In train.py:**
```python
import torch
from torch.distributed import init_process_group
from transformers import Trainer, TrainingArguments

# Trainer automatically handles DDP if multiple GPUs detected
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,  # Per GPU
    gradient_accumulation_steps=2,
    # Multi-GPU options
    ddp_find_unused_parameters=False,  # True if model has unused params
    gradient_checkpointing=True,        # Reduce memory
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()  # Automatically uses all GPUs
```

**Key Points:**
- Batch size multiplies by number of GPUs
- Training speed nearly linear with GPU count
- Requires NCCL for communication
- Set `CUDA_VISIBLE_DEVICES` to use specific GPUs

---

### Monitoring and Logging

```python
from transformers import TrainingArguments, Trainer
import wandb

# Login to Weights & Biases for free logging
wandb.login()

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    # Logging
    logging_steps=100,
    logging_dir="./logs",
    log_level="info",
    report_to=["wandb"],  # Log to W&B
    # Checkpointing
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,  # Keep only 3 best
    # Evaluation
    eval_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[
        # Early stopping
        transformers.EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.0
        )
    ]
)
trainer.train()
```

**Weights & Biases Dashboard:**
- Real-time loss curves
- System metrics (GPU, memory)
- Hyperparameter comparison
- Model versioning
- Free for open source

---

### Hyperparameter Tuning

**Grid Search:**
```python
# Manual approach - try different values
learning_rates = [1e-5, 3e-5, 5e-5]
batch_sizes = [8, 16, 32]

for lr in learning_rates:
    for bs in batch_sizes:
        training_args = TrainingArguments(
            learning_rate=lr,
            per_device_train_batch_size=bs,
            output_dir=f"./results/lr_{lr}_bs_{bs}",
        )
        trainer = Trainer(model=model, args=training_args, ...)
        trainer.train()
```

**Ray Tune Integration:**
```python
from transformers.integrations import TuneReportCallback

def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4),
        "per_device_train_batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "num_train_epochs": trial.suggest_int("num_epochs", 1, 5),
    }

best_trial = trainer.hyperparameter_search(
    direction="minimize",
    backend="ray",
    hp_space=hp_space,
    n_trials=10,
)
```

---

### Validation Strategies

```python
from datasets import load_metric

metric = load_metric("accuracy")  # For classification
# Or: "bleu", "rouge", "f1", "exact_match"

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,  # Called during eval
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
```

**Key Metrics by Task:**
- **Classification:** Accuracy, F1, Precision, Recall
- **Generation:** BLEU, ROUGE, METEOR
- **QA:** Exact Match, F1
- **Language Modeling:** Perplexity

---

## Advanced Level

### QLoRA for 7B Models on Consumer GPUs

**Complete QLoRA Setup:**

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# 1. Setup 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NF4 (better than int4)
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,      # Double quantization for even less memory
)

# 2. Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    quantization_config=bnb_config,
    device_map="auto",  # Automatically distributed across GPUs/CPU
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")
tokenizer.pad_token = tokenizer.eos_token

# 3. Prepare for k-bit training (freezes non-LoRA weights)
model = prepare_model_for_kbit_training(model)

# 4. Setup LoRA
lora_config = LoraConfig(
    r=16,                              # Rank
    lora_alpha=32,                      # Scaling
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# 5. Training with SFTTrainer (designed for instruction tuning)
training_args = TrainingArguments(
    output_dir="./qlora-7b",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size: 16
    learning_rate=2e-4,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    max_grad_norm=0.3,
    optim="paged_adamw_32bit",  # Memory-efficient optimizer
)

# Dataset in messages format
from datasets import Dataset

data = [
    {
        "text": "### Instruction:\nWhat is 2+2?\n\n### Response:\n4"
    },
    # ... more examples
]
dataset = Dataset.from_list(data)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,
    packing=True,  # Pack multiple examples into single sequence
)

trainer.train()

# 6. Save adapter only (very small)
model.save_pretrained("./qlora-adapter")

# 7. Inference with merged model (optional)
from peft import PeftModel

# Load base model normally (or quantized)
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    torch_dtype=torch.float16,
    device_map="auto",
)

# Load adapter
model = PeftModel.from_pretrained(base_model, "./qlora-adapter")

# Optional: merge and unload (creates full model, requires space)
model = model.merge_and_unload()  # Now you can save as regular model
model.save_pretrained("./merged-model")
```

**Memory Usage:**
- Base 7B model: ~14GB fp32, ~7GB fp16
- Quantized: ~2GB
- LoRA adapters: ~10MB
- Total with QLoRA: ~2GB (can fit on T4!)

---

### DPO Training

**Complete DPO Implementation:**

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset

# 1. Load base model (ideally SFT model, but pretrained works)
model_name = "meta-llama/Llama-2-7b"
model = AutoModelForCausalLM.from_pretrained(model_name)
ref_model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 2. Prepare preference dataset
# Format: {"prompt": [...], "chosen": [...], "rejected": [...]}
preference_data = [
    {
        "prompt": [{"role": "user", "content": "What is 2+2?"}],
        "chosen": [{"role": "assistant", "content": "2+2 equals 4."}],
        "rejected": [{"role": "assistant", "content": "2+2 equals 5."}],
    },
    # ... more examples
]

train_dataset = Dataset.from_list(preference_data)

# 3. Configure DPO
dpo_config = DPOConfig(
    output_dir="./dpo-model",
    beta=0.1,                          # KL divergence penalty (0.1-0.5)
    learning_rate=5e-7,                # Much lower than SFT
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    max_prompt_length=512,
    max_length=1024,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
)

# 4. Train with DPOTrainer
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# 5. Save and evaluate
trainer.save_model("./dpo-aligned-model")

# 6. Inference
def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0])

print(generate("What is 2+2?"))
```

**DPO Tips:**
- Beta controls how much model can deviate from reference
  - Low beta (0.05): Conservative, stays close to reference
  - High beta (0.5): Aggressive, optimizes heavily for preference
- Start with beta=0.1
- Learning rate should be 10-100x lower than SFT
- Requires high-quality preference data (diverse, balanced)

---

### RLHF Implementation (Advanced)

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch

# Stage 1: Use pre-trained SFT model (or train your own)
sft_model_path = "./sft-model"

# Stage 2: Reward model (simplified - use trained reward model in practice)
class SimpleRewardModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.head = torch.nn.Linear(base_model.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.base(input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state[:, -1, :]
        return self.head(hidden)

# Stage 3: PPO Training
ppo_config = PPOConfig(
    model_name=sft_model_path,
    learning_rate=1.41e-5,
    log_with="wandb",
    batch_size=32,
)

model = AutoModelForCausalLMWithValueHead.from_pretrained(sft_model_path)
ref_model = AutoModelForCausalLM.from_pretrained(sft_model_path)
tokenizer = AutoTokenizer.from_pretrained(sft_model_path)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)

# Load reward model (pre-trained on preference data)
reward_model = SimpleRewardModel(AutoModelForCausalLM.from_pretrained(sft_model_path))
reward_model.eval()

# Generate prompts and train
dataset = load_dataset("your-prompt-dataset")

for epoch in range(num_epochs):
    for batch in dataset:
        # Generate responses
        query_tensors = tokenizer.encode(batch["prompt"], return_tensors="pt")
        response_tensors = ppo_trainer.generate(query_tensors)
        
        # Get rewards
        with torch.no_grad():
            rewards = reward_model(response_tensors)
        
        # PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        if stats['ppo/loss/total'] > 100:  # Check for divergence
            print("Training diverged! Reduce learning rate.")
            break

model.save_pretrained("./rlhf-aligned-model")
```

---

### Distributed Training with DeepSpeed

**Setup for 8 GPU training:**

```bash
# Install DeepSpeed
pip install deepspeed

# Run with 8 GPUs
deepspeed --num_gpus=8 train.py --deepspeed deepspeed_config.json
```

**deepspeed_config.json:**
```json
{
    "train_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 5e-5
        }
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }
    },
    "fp16": {
        "enabled": true,
        "loss_scale_window": 1000
    }
}
```

**train.py:**
```python
import deepspeed
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    deepspeed="deepspeed_config.json",
    per_device_train_batch_size=4,
    # ... other args
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
```

---

### Production Deployment of Fine-Tuned Models

**Option 1: Merge and Deploy Full Model**
```python
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
lora = PeftModel.from_pretrained(base, "./lora-weights")
merged = lora.merge_and_unload()
merged.save_pretrained("./production-model")

# Push to Hub
merged.push_to_hub("username/my-model")
```

**Option 2: Deploy with Adapter (Smaller)**
```python
# Just save adapter
adapter.save_pretrained("./adapters/my-lora")

# In inference
base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
model = PeftModel.from_pretrained(base, "./adapters/my-lora")
model.eval()
```

**Option 3: Inference Server**
```bash
# Using vLLM (optimized inference)
pip install vllm

# Start server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b \
    --load-format safetensors \
    --tensor-parallel-size 1

# Query
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-2-7b",
        "prompt": "What is AI?",
        "max_tokens": 100
    }'
```

---

## Training Formats & Data Preparation

### JSONL Format (Instruction Tuning)

**Standard instruction format:**
```jsonl
{"instruction": "Classify sentiment", "input": "I love this!", "output": "positive"}
{"instruction": "Summarize", "input": "Long text...", "output": "Summary..."}
```

**Conversation format (multi-turn):**
```jsonl
{
  "conversations": [
    {"from": "human", "value": "What is 2+2?"},
    {"from": "gpt", "value": "2+2 equals 4."}
  ]
}
```

**Messages format (OpenAI style):**
```jsonl
{
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4."}
  ]
}
```

### CSV Format

**Simple classification:**
```csv
text,label
"I love this!",positive
"This is terrible",negative
"It's okay",neutral
```

**Instruction pairs:**
```csv
instruction,input,output
"Classify sentiment","I love this","positive"
"Summarize","Long text...","Summary..."
```

### Conversational Format

**Chat data structure:**
```jsonl
{
  "prompt": "What is machine learning?",
  "chosen": "Machine learning is...",
  "rejected": "ML is just algorithms"
}
```

### Data Quality Guidelines

1. **Diversity:** Cover multiple domains, styles, difficulty levels
2. **Accuracy:** Remove factually incorrect examples
3. **Clarity:** Instructions should be unambiguous
4. **Format Consistency:** All examples follow same template
5. **Balance:** Equal representation across classes/tasks
6. **Size Recommendations:**
   - Classification: 100-1000 per class
   - Generation: 1000-10000 examples
   - Instruction tuning: 5000-50000 diverse instructions
   - Larger models benefit from more data

### Data Annotation Guidelines

**For preference pairs (RLHF/DPO):**
1. Have multiple annotators rate each pair
2. Calculate inter-annotator agreement (Cohen's kappa)
3. Discard low-agreement examples
4. Clear rubric defining what "better" means
5. Balance: ~50% wins for each response in pairs

**For instructions:**
1. Clear, specific instructions
2. Examples with expected outputs
3. Diverse task types
4. Include edge cases and challenging examples
5. Human review before training

---

## Training Infrastructure

### Hardware Requirements by Method

| Method | 7B Model | 13B Model | 70B Model |
|--------|----------|-----------|-----------|
| Full FT | 24GB+ | 40GB+ | 160GB+ |
| LoRA | 16GB | 24GB | 48GB |
| QLoRA | 8GB | 12GB | 24GB |
| DPO | 16GB | 24GB | 48GB |

### Cloud Providers

**Hugging Face Spaces (Free)**
- Limited GPU (T4, no GPU option)
- Free tier: 50GB storage
- Paid: $5-50/month for better GPUs

**Google Colab**
- Free: T4 GPU (16GB), ~12 hours
- Pro: A100 GPU (40GB), unlimited
- Cost: $10/month

**Lambda Labs**
- Spot: $0.30/hr (RTX 6000 48GB)
- On-demand: $1-2/hr
- Great for short training runs

**Vast.ai**
- Spot pricing: Often $0.10-0.50/hr
- Access to diverse GPUs
- Manual instance management required

**Hugging Face on Cloud**
- Auto-scaling infrastructure
- Pay per hour
- Integrated with Hub

### Local Setup with Docker

```dockerfile
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip git

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install transformers datasets peft trl bitsandbytes accelerate

WORKDIR /workspace
```

**Run:**
```bash
docker build -t train-hf .
docker run --gpus all -v $(pwd):/workspace train-hf python train.py
```

### Environment Setup (venv)

```bash
# Create environment
python -m venv hf-train
source hf-train/bin/activate  # On Windows: hf-train\Scripts\activate

# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate

# Install for LoRA/QLoRA
pip install peft bitsandbytes

# Install for advanced training
pip install trl wandb deepspeed

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Common Training Frameworks

### Hugging Face Trainer (Recommended for Beginners)

**Pros:**
- Easiest to use
- Handles distributed training automatically
- Built-in evaluation, logging, checkpointing
- Integrates with Hub

**Cons:**
- Less flexible for custom training loops
- Slower than hand-optimized PyTorch

---

### TRL (Transformers Reinforcement Learning)

**Best for:**
- DPO training
- PPO/RLHF
- Preference-based alignment

**Key Classes:**
- `SFTTrainer`: Supervised fine-tuning
- `DPOTrainer`: Direct preference optimization
- `PPOTrainer`: Proximal policy optimization

---

### PyTorch Lightning

**Best for:**
- Complex custom training loops
- Research/experimental training

```python
import pytorch_lightning as pl

class LLMTuner(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        return outputs.loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=5e-5)

trainer = pl.Trainer(gpus=1, max_epochs=3)
trainer.fit(model, train_dataloader)
```

---

### Accelerate (For Custom Loops)

```python
from accelerate import Accelerator

accelerator = Accelerator()

model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)

for epoch in range(num_epochs):
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
```

---

### Ollama (Local Training/Serving)

```bash
# Train
ollama run llama2 "Fine-tune me on your data"

# Serve
ollama serve

# Query
curl http://localhost:11434/api/generate -d '{
  "model": "llama2",
  "prompt": "What is AI?"
}'
```

---

## Datasets

### Where to Find Datasets

**Hugging Face Hub (Best Source)**
- 100,000+ public datasets
- Search by task: nlp, language-modeling, instruction-tuning
- Direct loading with `datasets` library

**Popular Instruction Tuning Datasets:**
1. **Alpaca** (52k examples)
   - High-quality instruction-following
   - Format: instruction, input, output
   - ```python
     from datasets import load_dataset
     dataset = load_dataset("tatsu-lab/alpaca")
     ```

2. **Dolly** (15k examples)
   - Open-source alternative to Alpaca
   - Diverse tasks and categories

3. **OpenAssistant** (161k conversations)
   - Multi-turn conversations
   - Highly diverse

4. **WizardLM** (70k examples)
   - High-complexity instructions
   - Better for advanced reasoning

5. **UltraChat** (1.4M conversations)
   - Large-scale conversations
   - Good for general instruction following

**Preference/Alignment Datasets:**
1. **HH-RLHF** (169k examples)
   - Helpfulness vs Harmlessness
   - Good for DPO/RLHF training

2. **UltraFeedback** (64k examples)
   - GPT-4 generated feedback
   - Multiple ranking dimensions

3. **OpenAssistant Preferences** (10k examples)
   - Human preference rankings

### How to Prepare Your Own Dataset

**Step 1: Collection**
```python
# From various sources
data = [
    {
        "instruction": "Explain machine learning",
        "output": "Machine learning is..."
    },
    # ... more examples
]
```

**Step 2: Cleaning**
```python
# Remove duplicates
unique = {}
for example in data:
    key = (example["instruction"], example["output"])
    if key not in unique:
        unique[key] = example

# Remove invalid examples
cleaned = [ex for ex in unique.values() if len(ex["output"]) > 10]

# Remove PII
import re
for ex in cleaned:
    ex["output"] = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', ex["output"])
```

**Step 3: Format**
```python
from datasets import Dataset

formatted = []
for ex in cleaned:
    formatted.append({
        "instruction": ex["instruction"],
        "input": ex.get("input", ""),
        "output": ex["output"]
    })

dataset = Dataset.from_list(formatted)
```

**Step 4: Split**
```python
# 80-10-10 split
split = dataset.train_test_split(test_size=0.2, seed=42)
train_val = split["train"].train_test_split(test_size=0.125, seed=42)

train_dataset = train_val["train"]
eval_dataset = train_val["test"]
test_dataset = split["test"]
```

**Step 5: Save**
```python
dataset.save_to_disk("./my-dataset")
# Or push to Hub
dataset.push_to_hub("username/my-dataset")
```

### Data Licensing Considerations

- **CC0:** Public domain, use freely
- **CC-BY:** Credit the author
- **CC-BY-SA:** Credit and share-alike
- **MIT:** Use commercially, credit author
- **OpenAI Restricted:** Cannot use to train competing models
- Always check dataset license before use

---

## Quick Reference: Method Selection

**Choose Full Fine-Tuning if:**
- You have 24GB+ VRAM
- You want maximum quality
- You need complete model control

**Choose LoRA if:**
- You have 16GB VRAM
- You want fast training
- You need small checkpoints (< 1GB)

**Choose QLoRA if:**
- You have 8-12GB VRAM
- You're training on consumer hardware
- Cost is critical

**Choose DPO if:**
- You have preference data (better/worse responses)
- You want alignment without RLHF complexity
- You want stable training

**Choose RLHF if:**
- You have preference data at scale
- You have substantial compute (3+ GPUs)
- You need production-grade alignment

---

## Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| CUDA OOM | Batch too large | Reduce `per_device_train_batch_size` |
| NaN loss | Learning rate too high | Reduce learning rate 10x |
| Slow training | No gradient checkpointing | Add `gradient_checkpointing=True` |
| High memory | Large model size | Use QLoRA or smaller model |
| Poor results | Bad data | Check dataset quality, increase size |
| Training diverges | Hyperparameter issue | Reduce learning rate, check data |

---

## Additional Resources

**Official Documentation:**
- Hugging Face Transformers: https://huggingface.co/docs/transformers
- PEFT Library: https://huggingface.co/docs/peft
- TRL: https://huggingface.co/docs/trl

**Papers:**
- LoRA: https://arxiv.org/abs/2106.09685
- QLoRA: https://arxiv.org/abs/2305.14314
- DPO: https://arxiv.org/abs/2305.18290
- InstructGPT (RLHF): https://arxiv.org/abs/2203.02155

**Recommended Courses:**
- Hugging Face Course: https://huggingface.co/course
- DeepLearning.AI RLHF: https://learn.deeplearning.ai

---

**Last Updated:** April 2026
**Versions:** Transformers 4.40+, PyTorch 2.0+, PEFT 0.9+
