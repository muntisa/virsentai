# Fine-Tuning Training Pipeline

This document describes the fine-tuning pipeline for the VirSeNtAI model based on HyenaDNA.

## Overview

The pipeline consists of three sequential fine-tuning scripts that progressively improve the model:

| Version | Script | Description |
|---------|--------|-------------|
| v1 | `FT-virsentai_v3_1.py` | Fine-tune original HyenaDNA model from HuggingFace |
| v2 | `FT-virsentai_v3_2.py` | Improved model with better checkpointing |
| v3 | `FT-virsentai_v3_3.py` | Further improvements with enhanced training |

## Prerequisites

- Python 3.x with .venv virtual environment
- Required packages: transformers, torch, datasets, sklearn, safetensors
- Pre-trained HyenaDNA model from HuggingFace
- Training and validation datasets in TSV format

## Pipeline Steps

### Step 1: Fine-Tune Original HyenaDNA Model

```powershell
python FT-virsentai_v3_1.py
```

**What it does:**
- Loads the original HyenaDNA model from HuggingFace
- Fine-tunes on training data (train_split_160k.tsv)
- Validates on validation data (val_split_160k.tsv)
- Converts model from .bin to safetensors format

**Input:**
- `fine-tuning/train_split_160k.tsv`
- `fine-tuning/val_split_160k.tsv`

**Output:**
- `fine-tuning/models/hyena_fine_tuned_v1/`

**Key Features:**
- safe_state_dict() to handle shared/tied tensors in HyenaDNA
- Comprehensive metrics tracking (accuracy, AUC, F1, etc.)
- Early stopping to prevent overfitting

---

### Step 2: Improved Model Checkpointing

```powershell
python FT-virsentai_v3_2.py
```

**What it does:**
- Loads the pre-trained model from v1
- Fine-tunes on the same dataset with improved settings
- Implements better model checkpointing

**Input:**
- `fine-tuning/train_split_160k.tsv`
- `fine-tuning/val_split_160k.tsv`
- Pre-trained model from v1 (optional)

**Output:**
- `fine-tuning/models/hyena_fine_tuned_v2/`

**Key Features:**
- Improved model saving
- Enhanced evaluation metrics
- Early stopping with patience

---

### Step 3: Further Improvements

```powershell
python FT-virsentai_v3_3.py
```

**What it does:**
- Loads the pre-trained model from v2
- Fine-tunes with enhanced training arguments
- Optimizes for better performance

**Input:**
- `fine-tuning/train_split_160k.tsv`
- `fine-tuning/val_split_160k.tsv`
- Pre-trained model from v2 (optional)

**Output:**
- `fine-tuning/models/hyena_fine_tuned_v3/`

**Key Features:**
- Enhanced training arguments
- Improved learning rate scheduling
- Best model saving based on AUROC
- AdamW optimizer with weight decay
- Confusion matrix tracking

## Metrics Tracked

All versions track the following metrics:

| Metric | Description |
|--------|-------------|
| Accuracy | Classification accuracy |
| AUC | Area Under ROC Curve |
| Precision | Positive predictive value |
| Recall | True positive rate |
| F1 | Harmonic mean of precision and recall |
| Average Precision | Area Under Precision-Recall Curve |
| Confusion Matrix | True/False Positives/Negatives |

## Model Format

All models are saved in **safetensors** format for:
- Better security (memory-safe)
- Faster loading
- Smaller file size

The `safe_state_dict()` function handles shared/tied tensors common in HyenaDNA architecture.

## Training Configuration

Common training settings across all versions:
- Early stopping with patience
- Learning rate scheduling
- Batch size and gradient accumulation
- Warmup steps for stable training

## Logs

All training runs generate logs saved to the `logs/` directory with timestamps.

## Summary

The pipeline progresses from the base HyenaDNA model through three iterations of improvement, each adding:
- Better model checkpointing
- Enhanced training arguments
- Comprehensive evaluation metrics

The final model (`hyena_fine_tuned_v3`) represents the best-performing version for viral genome classification.