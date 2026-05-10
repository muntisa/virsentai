#!/usr/bin/env python3
"""
Virus Prediction Script

Uses a fine-tuned HyenaDNA model to make predictions on virus sequences
from any TSV file containing a 'sequence' column.

Usage:
    python 02_predict_NEW.py --input path/to/my_sequences.tsv

Input:
    - Input TSV file with 'sequence' column (required)
    - Fine-tuned model weights (from FT-virsentai_v3 scripts)

Output:
    - ds/ds_160k_UNK_predictions.tsv (or similar based on input)
    - All original columns preserved + new column: PClass_1

Process:
    1. Load original HyenaDNA architecture from HuggingFace
    2. Copy best model weights from fine-tuned model
    3. Load sequences from input TSV file
    4. Tokenize sequences (max length: SEQ_MAX_LENGTH)
    5. Run inference
    6. Apply softmax to get probabilities
    7. Extract probability of class 1 (zoonotic potential)
    8. Save predictions to output TSV file

Requirements:
    - Input TSV with a 'sequence' column
    - Fine-tuned model in fine-tuning/models/ directory
"""

import os
import csv
import logging
import argparse
from datetime import datetime

import pandas as pd
csv.field_size_limit(2000000)

import torch
from scipy.special import softmax
from safetensors.torch import load_file
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

from config import *

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
if not hasattr(__builtins__, 'SIGALRM'):
    try:
        import signal
        signal.SIGALRM = 14
    except Exception:
        pass

# --- Configuration --------------------
CHECKPOINT = "LongSafari/hyenadna-medium-160k-seqlen-hf"
BEST_MODEL_PATH = "fine-tuning/models/virsentai3/backup-checkpoint-600"
MAX_LENGTH = 160_000

parser = argparse.ArgumentParser(
    description="Predict zoonotic potential from a TSV file with a 'sequence' column."
)
parser.add_argument(
    "--input",
    required=True,
    help="Path to the input TSV file. Must contain a 'sequence' column.",
)
args = parser.parse_args()

input_tsv_path = os.path.abspath(args.input)

input_dir  = os.path.dirname(input_tsv_path)
input_stem = os.path.splitext(os.path.basename(input_tsv_path))[0]
output_tsv_path = os.path.join(input_dir, f"{input_stem}_predictions.tsv")

tokenized_cache_dir = os.path.join(UNK_TOKENIZED_DIR, f"{input_stem}_tokenized")

script_name   = os.path.splitext(os.path.basename(__file__))[0]
timestamp     = datetime.now().strftime("%Y%m%d_%H%M%SS")
os.makedirs(LOG_DIR, exist_ok=True)
log_file_name = os.path.join(LOG_DIR, f"{script_name}_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_name),
        logging.StreamHandler(),
    ],
)


def predict_with_model(input_path: str, output_path: str) -> None:
    try:
        logging.info("Starting prediction process...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        logging.info(f"Loading tokenizer from {CHECKPOINT}...")
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)

        logging.info(f"Loading model architecture from {CHECKPOINT}...")
        model = AutoModelForSequenceClassification.from_pretrained(
            CHECKPOINT,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        ).to(device)

        logging.info(f"Loading best model weights from {BEST_MODEL_PATH}...")
        state_dict = load_file(os.path.join(BEST_MODEL_PATH, "model.safetensors"))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logging.warning(f"Missing keys: {missing}")
        if unexpected:
            logging.warning(f"Unexpected keys: {unexpected}")
        logging.info("[OK] Model weights loaded successfully.")

        model.eval()
        model.config.use_cache = True
        logging.info("Model set to eval() mode.")

        logging.info(f"Loading sequences from: {input_path}")
        df = pd.read_csv(input_path, sep="\t")

        if "sequence" not in df.columns:
            raise ValueError(
                f"Input file must contain a 'sequence' column. "
                f"Found columns: {list(df.columns)}"
            )
        logging.info(f"Loaded {len(df):,} sequences.")

        logging.info("Tokenizing sequences...")
        dataset = Dataset.from_pandas(df[["sequence"]])

        def tokenize_function(examples):
            return tokenizer(
                examples["sequence"],
                truncation=True,
                padding="max_length",
                max_length=SEQ_MAX_LENGTH,
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["sequence"],
        )
        tokenized_dataset.set_format("torch")

        training_args = TrainingArguments(
            output_dir=os.path.join(BEST_MODEL_PATH, PREDICT_TEMP_SUBFOLDER),
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            bf16=True,
            report_to="none",
        )
        trainer = Trainer(model=model, args=training_args)

        torch.cuda.empty_cache()
        logging.info("Running inference...")
        predictions = trainer.predict(tokenized_dataset)

        logits = predictions.predictions
        probabilities = softmax(torch.from_numpy(logits).float().numpy(), axis=1)
        df["PClass_1"] = probabilities[:, 1]

        logging.info(f"Saving predictions to: {output_path}")
        df.to_csv(output_path, sep="\t", index=False)
        logging.info(f"Done. {len(df):,} predictions saved.")

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except ValueError as e:
        logging.error(str(e))
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":

    if not os.path.exists(input_tsv_path):
        print(f"Error: Input file not found: {input_tsv_path}")
        exit(1)

    if os.path.exists(output_tsv_path):
        print(f"Output file already exists: {output_tsv_path}")
        print("Delete it to re-run predictions.")
        exit(0)

    predict_with_model(input_tsv_path, output_tsv_path)

    if os.path.exists(output_tsv_path):
        print("\n--- Verification ---")
        df_check = pd.read_csv(output_tsv_path, sep="\t")
        print(f"Rows: {len(df_check):,}  |  Columns: {list(df_check.columns)}")
        print(df_check.head(2).to_string())