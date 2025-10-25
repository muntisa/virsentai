# %% [markdown]
# VirSentAI: Viral Sentry AI – Intelligent Zoonotic Surveillance Platform
# Training HyenaDNA model 160k classifier for viruses host: Human vs. nonHuman (zoonotic viruses)
#
# This script fine-tunes a pre-trained HyenaDNA model for DNA sequence classification.
# The key improvements in this version include:
# - **Disk Caching for Tokenized Data**: Saves tokenized datasets to disk to avoid re-running tokenization.
# - **Gradient Accumulation**: To simulate a larger batch size and stabilize training.
# - **Adaptive Optimizer**: Using AdamW, which is well-suited for transformer models.
# - **Learning Rate Scheduling**: A linear warmup and decay schedule to improve convergence.
# - **Increased Epochs**: More training epochs to allow the model to learn more thoroughly.
# - **Best Model Saving**: The trainer will now monitor the AUROC metric and save the best-performing model.

# Pre-trained model:
# https://huggingface.co/LongSafari/hyenadna-medium-160k-seqlen-hf

# %%
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback
import torch
from datasets import Dataset, load_from_disk
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import os
import logging
import tqdm as notebook_tqdm

from config import * # import global variables (files, folders, params, etc.)

# %%
print(">> VirSentAI training using pre-trained Hyenadna ...")

# Check for CUDA availability and set the device
print(f"CUDA Available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Instantiate pretrained model and tokenizer
checkpoint = 'LongSafari/hyenadna-medium-160k-seqlen-hf' # pre-trained model from HuggingFace
max_length = 160_000

# Using bfloat16 for better speed and reduced memory usage during training
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    torch_dtype=torch.bfloat16,
    device_map="auto",  # Automatically uses available GPUs
    trust_remote_code=True
).to(device)
print("Model and tokenizer loaded successfully.")

# %%
# Load data from CSV
print("Loading data from CSV...")
df = pd.read_csv(DATASET_FILE) # your dataset
print(f"Data loaded with {len(df)} rows and {len(df.columns)} columns.")

# %% [markdown]
# ### SKIP this cell if you don't want to use only a sample of the dataset!!!!!!

# %%
testing = False  # Set to True if you want to run a quick test with a smaller dataset

if testing:
    print("Running in TESTING mode with a 1% sample of the data.")
    # For testing purposes, sample a smaller subset of the data
    # Separate dataframes for each class to ensure balanced sampling
    df_class_0 = df[df['Class'] == 0]
    df_class_1 = df[df['Class'] == 1]

    # Sample 1% from each class
    df_class_0_sampled = df_class_0.sample(frac=0.01, random_state=2025)
    df_class_1_sampled = df_class_1.sample(frac=0.01, random_state=2025)

    # Concatenate and shuffle the sampled dataframes
    df_sampled = pd.concat([df_class_0_sampled, df_class_1_sampled])
    df = df_sampled.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Sampled data head:")
    print(df.head())

# %% [markdown]
# ### Split and Tokenize the Dataset (with Caching)

# %%
# --- Define paths for saving/loading tokenized datasets ---
# tokenize only one time for more training
tokenized_train_path = TOKEN_TRAIN_PATH
tokenized_val_path   = TOKEN_VAL_PATH

if os.path.exists(tokenized_train_path) and os.path.exists(tokenized_val_path):
    # --- Load pre-tokenized datasets from disk ---
    print(f"Loading tokenized datasets from {tokenized_train_path} and {tokenized_val_path}...")
    tokenized_train_dataset = load_from_disk(tokenized_train_path)
    tokenized_val_dataset = load_from_disk(tokenized_val_path)
    print("Datasets loaded successfully from disk.")

else:
    # --- Tokenize and save datasets if not found on disk ---
    print("Pre-tokenized datasets not found. Starting tokenization process...")
    
    # Split data into train and validation sets
    print("Splitting data into training and validation sets...")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=2025, stratify=df['Class'])

    # Tokenize sequences
    def tokenize_function(examples):
        """Tokenizes sequences using the pre-trained tokenizer."""
        return tokenizer(
            examples["seq"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    # Create Hugging Face Datasets
    print("Creating Hugging Face datasets...")
    train_dataset = Dataset.from_pandas(train_df[["Seq", "Class"]].rename(columns={"Seq": "seq", "Class": "labels"}))
    val_dataset = Dataset.from_pandas(val_df[["Seq", "Class"]].rename(columns={"Seq": "seq", "Class": "labels"}))

    # Tokenize the datasets in a batched manner for efficiency
    print("Tokenizing train dataset...")
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["seq"])
    print("Tokenizing validation dataset...")
    tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["seq"])

    # --- Save the tokenized datasets to disk for future use ---
    print(f"Saving tokenized datasets to {tokenized_train_path} and {tokenized_val_path}...")
    tokenized_train_dataset.save_to_disk(tokenized_train_path)
    tokenized_val_dataset.save_to_disk(tokenized_val_path)
    print("Datasets saved successfully.")


# Set format to PyTorch tensors for both loaded and newly created datasets
print("Setting dataset format to PyTorch...")
tokenized_train_dataset.set_format("torch")
tokenized_val_dataset.set_format("torch")
print("Datasets are ready for training.")

# %%
# Define a function to compute evaluation metrics
def compute_metrics(eval_pred):
    """Computes accuracy and AUROC for evaluation."""
    logits, labels = eval_pred
    # The output logits might be in a different format (e.g., bfloat16), so convert to float32 for stable computation
    logits = torch.from_numpy(logits).float().numpy()
    
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    
    # Calculate AUROC using the probabilities of the positive class
    # Apply softmax to logits to get probabilities
    softmax_logits = torch.nn.functional.softmax(torch.from_numpy(logits), dim=-1).numpy()
    auroc = roc_auc_score(labels, softmax_logits[:, 1])  # Assuming binary classification, positive class is index 1
    
    return {"accuracy": accuracy, "auroc": auroc}

# --- Enhanced Training Arguments ---
# Here we define the training parameters.
training_args = TrainingArguments(
    output_dir=TRAINING_DIR,
    
    # --- Epochs and Batch Size ---
    num_train_epochs=5,  # Increased epochs for more comprehensive training.
    per_device_train_batch_size=2,  # The batch size per GPU. Kept small due to model size.
    per_device_eval_batch_size=4,   # Can be larger than train batch size as no gradients are stored.

    # --- Gradient Accumulation ---
    # This is a key technique for training large models on limited hardware.
    # It accumulates gradients over several steps before performing an optimizer update.
    # Effective Batch Size = per_device_train_batch_size * gradient_accumulation_steps = 2 * 8 = 16
    gradient_accumulation_steps=8,
    
    # --- Optimizer and Learning Rate ---
    optim="adamw_torch",  # AdamW is an adaptive optimizer, great for transformers.
    learning_rate=2e-5,   # A standard learning rate for fine-tuning.
    lr_scheduler_type="linear", # Use a linear learning rate decay schedule.
    warmup_ratio=0.1,     # 10% of training steps will be used for a linear warmup.
    weight_decay=0.01,    # Adds L2 regularization.

    # --- Evaluation and Saving Strategy ---
    # NOTE: Renamed `evaluation_strategy` to `eval_strategy` to support older
    # versions of the `transformers` library. The argument `evaluation_strategy`
    # caused the TypeError.
    eval_strategy="epoch", # Evaluate at the end of each epoch.
    save_strategy="epoch",       # Save a checkpoint at the end of each epoch.
    load_best_model_at_end=True, # Load the best model (based on metric) at the end of training.
    metric_for_best_model="auroc", # Use AUROC to determine the best model.
    greater_is_better=True,      # A higher AUROC is better.

    # --- Memory and Performance ---
    fp16=False, # Set to True if using float16 precision, but we use bfloat16 via model dtype.
    bf16=True,  # Enable bfloat16 training for performance on compatible GPUs (Ampere or newer).
    gradient_checkpointing=True, # Saves memory by re-computing activations instead of storing them.

    # --- Logging and Reporting ---
    logging_strategy="steps",
    logging_steps=10,
    save_safetensors=False, # Disable safetensors as requested.
    report_to="tensorboard", # Log to tensorboard. Can be changed to "wandb" or "none".
)

# Set logging level
logging.basicConfig(level=logging.INFO)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)] # Stop if auroc doesn't improve.
)

# %%
# Clear GPU memory before starting training
torch.cuda.empty_cache()

print("Starting model training...")
train_result = trainer.train()

print("Training finished.")
print(train_result)

# Save the final trained model
print("Saving the best trained model...")
output_model_path = NEW_MODEL_DIR
if not os.path.exists(output_model_path):
    os.makedirs(output_model_path)

trainer.save_model(output_model_path)
tokenizer.save_pretrained(output_model_path)
print(f"Trained model and tokenizer saved to {output_model_path}")

# %%
# --- Final Evaluation ---
# The trainer automatically loads the best model at the end,
# so we can directly evaluate its performance.

# Clear GPU memory
torch.cuda.empty_cache()

print("Evaluating final model on the validation set...")
val_results = trainer.evaluate(tokenized_val_dataset)
print("--- Validation Set Results ---")
for key, value in val_results.items():
    print(f"{key}: {value:.4f}")

# Evaluate on the training set to check for overfitting
print("\nEvaluating final model on the training set...")
train_results = trainer.evaluate(tokenized_train_dataset)
print("--- Training Set Results ---")
for key, value in train_results.items():
    print(f"{key}: {value:.4f}")

