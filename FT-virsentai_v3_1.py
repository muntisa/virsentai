#!/usr/bin/env python3
"""
Fine-Tuning Script v1 - Original HyenaDNA Model

Fine-tunes the original HyenaDNA model from HuggingFace.
Converts the model from .bin format to safetensors format for better compatibility.

Usage:
    python FT-virsentai_v3_1.py

Requirements:
    - transformers, torch, datasets, sklearn, safetensors
    - Pre-trained HyenaDNA model from HuggingFace
    - Training and validation datasets (TSV format)
    - config.py with paths

Input:
    - fine-tuning/train_split_160k.tsv (training data)
    - fine-tuning/val_split_160k.tsv (validation data)

Output:
    - fine-tuning/models/virsentai/ (model files in safetensors format)
    - Logs saved to logs/

Notes:
    - Uses safe_state_dict() to handle shared/tied tensors in HyenaDNA
    - Converts .bin to .safetensors format before saving
    - Implements comprehensive metrics tracking (accuracy, AUC, F1, etc.)
    - Uses early stopping to prevent overfitting
"""

import time
from datetime import datetime, timedelta
import os
import sys
import csv
import json
import logging

import numpy as np
import pandas as pd

import torch
from torch import nn

from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    EarlyStoppingCallback, 
    TrainerCallback
)
from datasets import Dataset, load_from_disk
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, average_precision_score, confusion_matrix
)
from safetensors.torch import save_file, load_file
        
csv.field_size_limit(2000000)

def safe_state_dict(model):
    """
    safetensors refuses to save shared/tied tensors (common in Hyena).
    This clones any tensor whose storage is already seen, so every key
    gets its own unique backing memory before we call save_file().
    """
    sd = model.state_dict()
    seen_ptrs: dict[int, str] = {}
    out: dict[str, torch.Tensor] = {}
    for key, tensor in sd.items():
        ptr = tensor.data_ptr()
        if ptr in seen_ptrs:
            out[key] = tensor.clone()   # break the sharing
        else:
            seen_ptrs[ptr] = key
            out[key] = tensor
    return out
            
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits_tensor = torch.from_numpy(logits).float()
    predictions = np.argmax(logits, axis=-1)
    probs = torch.nn.functional.softmax(logits_tensor, dim=-1).numpy()
    prob_positive = probs[:, 1]  
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "auroc": roc_auc_score(labels, prob_positive),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "avg_precision": average_precision_score(labels, prob_positive),
    }

class HyenaCheckpointCallback(TrainerCallback):
    def __init__(self, save_steps=500, output_dir="./checkpoints", early_stopping_patience=8):
        self.save_steps = save_steps
        self.output_dir = output_dir
        self.best_auroc = 0.0
        self.best_model_path = os.path.join(output_dir, "best_model_manual")
        self.early_stopping_patience = early_stopping_patience
        self.patience_counter = 0                  # ← add this

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.save_steps == 0:
            path = os.path.join(self.output_dir, f"backup-checkpoint-{state.global_step}")
            os.makedirs(path, exist_ok=True)
            save_file(safe_state_dict(kwargs['model']), os.path.join(path, "model.safetensors"))
            kwargs['model'].config.save_pretrained(path)
            print(f"\n[Safety-Net] Backup saved at step {state.global_step}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_auroc" in metrics:
            current_auroc = metrics["eval_auroc"]
            if current_auroc > self.best_auroc:
                self.best_auroc = current_auroc
                self.patience_counter = 0          # ← reset on improvement
                os.makedirs(self.best_model_path, exist_ok=True)
                save_file(safe_state_dict(kwargs['model']), os.path.join(self.best_model_path, "model.safetensors"))
                kwargs['model'].config.save_pretrained(self.best_model_path)
                print(f"\n⭐ New Best Model! AUROC: {self.best_auroc:.4f}")
            else:
                self.patience_counter += 1         # ← increment on no improvement
                print(f"\n[Early Stop] No improvement for {self.patience_counter}/{self.early_stopping_patience} evals")
                if self.patience_counter >= self.early_stopping_patience:
                    print("\n🛑 Early stopping triggered.")
                    control.should_training_stop = True   # ← this is the key line
        return control

class MetricsLoggerCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.records = []
        self.train_start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start_time = time.time()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            elapsed = time.time() - self.train_start_time
            record = {
                "step":             state.global_step,
                "epoch":            round(state.epoch, 4),
                "elapsed_hours":    round(elapsed / 3600, 4),
                "eval_loss":        metrics.get("eval_loss"),
                "eval_auroc":       metrics.get("eval_auroc"),
                "eval_accuracy":    metrics.get("eval_accuracy"),
                "eval_precision":   metrics.get("eval_precision"),
                "eval_recall":      metrics.get("eval_recall"),
                "eval_f1":          metrics.get("eval_f1"),
                "eval_avg_precision": metrics.get("eval_avg_precision"),
            }
            self.records.append(record)
            print(f"\n[Metrics] Step {record['step']} | "
                  f"AUROC: {record['eval_auroc']:.4f} | "
                  f"Elapsed: {record['elapsed_hours']:.2f}h")

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.train_start_time
        os.makedirs(self.output_dir, exist_ok=True)

        # --- CSV ---
        csv_path = os.path.join(self.output_dir, "training_metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.records[0].keys())
            writer.writeheader()
            writer.writerows(self.records)

        # --- JSON (includes summary) ---
        best = max(self.records, key=lambda x: x["eval_auroc"] or 0)
        summary = {
            "total_training_hours": round(total_time / 3600, 4),
            "total_steps":          state.global_step,
            "total_epochs":         round(state.epoch, 4),
            "best_auroc":           best["eval_auroc"],
            "best_auroc_step":      best["step"],
            "best_auroc_epoch":     best["epoch"],
            "best_auroc_elapsed_hours": best["elapsed_hours"],
            "evals": self.records
        }
        json_path = os.path.join(self.output_dir, "training_metrics.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n📊 Metrics saved to {csv_path}")
        print(f"📊 Summary saved to {json_path}")
        print(f"⏱️  Total training time: {total_time/3600:.2f}h")
        print(f"⭐ Best AUROC: {best['eval_auroc']:.4f} at step {best['step']} "
              f"(epoch {best['epoch']}, {best['elapsed_hours']:.2f}h elapsed)")
 
# weighted cross-entropy to help class 1
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # Upweight positives slightly to close the precision/recall gap
        weight = torch.tensor([1.0, 1.3], device=logits.device, dtype=logits.dtype)
        loss = nn.CrossEntropyLoss(weight=weight)(logits, labels)
        return (loss, outputs) if return_outputs else loss

#####################################
# MAIN
#####################################     
if __name__ == "__main__":

    # =============================================================================
    # CONFIGURATION
    # =============================================================================

    TRAIN_FILE      = "fine-tuning/train_split_160k.tsv"
    VAL_FILE        = "fine-tuning/val_split_160k.tsv"
    TOKENIZED_TRAIN = "fine-tuning/tokenized_train_dataset"
    TOKENIZED_VAL   = "fine-tuning/tokenized_val_dataset"

    CHECKPOINT  = 'LongSafari/hyenadna-medium-160k-seqlen-hf'
    MAX_LENGTH  = 160_000

    # 2. New output dir
    OUTPUT_PATH = "fine-tuning/models/virsentai"
    
    start = time.perf_counter()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Architecture
    print(f"Loading architecture from {CHECKPOINT}...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    ).to(device)
    
    # no loaded weights

    # 3. Load Datasets
    if os.path.exists(TOKENIZED_TRAIN) and os.path.exists(TOKENIZED_VAL):
        tokenized_train_dataset = load_from_disk(TOKENIZED_TRAIN)
        tokenized_val_dataset = load_from_disk(TOKENIZED_VAL)
    else:
        # (Dataset tokenization logic from your original script goes here)
        pass

    tokenized_train_dataset.set_format("torch")
    tokenized_val_dataset.set_format("torch")
    model.gradient_checkpointing_enable()

    # 4. Corrective Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_PATH,
        num_train_epochs=2,
        optim="adamw_torch_fused",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        
        # Effective Batch Size = 16
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8, 
        per_device_eval_batch_size=4,
        
        learning_rate=2e-5,    # original value previous training
        warmup_ratio=0.1,      # Give it a bit more warmup to stabilize the larger LR
        max_grad_norm=1.0,     # back to 1.0, grad norms from scratch are more stable

        lr_scheduler_type="cosine",
        weight_decay=0.01,
        
        # Evaluation & Saving
        eval_strategy="epoch",         # ONLY evaluate at the end of an epoch (saves ~5 hours)
        
        save_strategy="no",             # Prevent the "Tied Weights" error
        load_best_model_at_end=False,   # We will do this manually now
        metric_for_best_model="auroc",
        greater_is_better=True,
        
        dataloader_num_workers=4,      # CRITICAL: Use your CPU cores to load data!
        dataloader_pin_memory=True,    # Faster data transfer for same GPU training
        logging_steps=50,
        report_to="tensorboard",
    )
    
    # no weights from other model!
    
    # Set logging level
    logging.basicConfig(level=logging.INFO)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            HyenaCheckpointCallback(
                save_steps=500,
                output_dir=OUTPUT_PATH,
                early_stopping_patience=8   # ← replaces EarlyStoppingCallback
            ),
            MetricsLoggerCallback(output_dir=OUTPUT_PATH)
        ]
    )

    print("Starting corrective training...")
    trainer.train()

    # 5. Load and save best model (NOT trainer.model which is the last state)
    print("Loading best model from checkpoint...")
    best_state = load_file(os.path.join(OUTPUT_PATH, "best_model_manual", "model.safetensors"))
    model.load_state_dict(best_state)

    print(f"Saving best model to {OUTPUT_PATH}...")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    save_file(safe_state_dict(model), os.path.join(OUTPUT_PATH, "model.safetensors"))
    trainer.model.config.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)
    print("✅ Best model saved (safetensors format).")

    end = time.perf_counter()
    seconds = end - start

    # Formats as H:MM:SS
    readable_time = str(timedelta(seconds=seconds))
    print(f"Execution time: {readable_time}")
    