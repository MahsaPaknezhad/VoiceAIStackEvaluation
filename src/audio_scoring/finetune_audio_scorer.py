#!/usr/bin/env python3
# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT
"""
Fine-tune MiniCPM-o 4.5 with LoRA on labeled audio samples.

Reads labels from: evaluation_data/training_dataset/labels/labels.json
Saves adapter to: trained_models/minicpm_audio_scorer_lora/
"""

import json
import logging
import os
from copy import deepcopy

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LABELS_FILE = "evaluation_data/training_dataset/labels/labels.json"
OUTPUT_DIR = "data/trained_models/minicpm_audio_scorer_lora"
MODEL_NAME = "openbmb/MiniCPM-o-4_5"

SCORING_PROMPT = (
    "You are an expert audio quality evaluator. Listen to this audio carefully and rate it on three dimensions:\n"
    "1. Naturalness (1-10): How natural and human-like does the speech sound?\n"
    "2. Noisiness (1-10): How noisy is the audio? 1=clean, 10=very noisy.\n"
    "3. Loudness (1-10): How loud is the audio? 1=barely audible, 10=very loud.\n\n"
    'Respond ONLY with JSON: {"naturalness": <int>, "noisiness": <int>, "loudness": <int>}'
)

MAX_EPOCHS = 50
LR = 1e-4
GRAD_ACCUM = 4
PATIENCE = 5
TRAIN_RATIO = 0.9
LOSS_PLOT_PATH = "data/trained_models/minicpm_audio_scorer_lora/training_loss.png"


def build_tokenized_sample(model, audio_array, target_json_str):
    """Use the model's processor to tokenize audio+prompt+target into input_ids."""
    msgs = [
        {"role": "user", "content": [SCORING_PROMPT, audio_array]},
        {"role": "assistant", "content": [target_json_str]},
    ]

    # Replicate what chat() does to build processor inputs
    copy_msgs = deepcopy(msgs)
    audios = []
    audio_parts = []
    for i, msg in enumerate(copy_msgs):
        content = msg["content"]
        cur_msgs = []
        for c in content:
            if isinstance(c, np.ndarray):
                audios.append(c)
                audio_parts.append(i)
                cur_msgs.append("<audio>./</audio>")
            elif isinstance(c, str):
                cur_msgs.append(c)
        msg["content"] = "\n".join(cur_msgs)

    prompt_text = model.processor.tokenizer.apply_chat_template(
        copy_msgs,
        tokenize=False,
        add_generation_prompt=False,
        use_tts_template=True,
    )

    inputs = model.processor(
        [prompt_text],
        [[]],       # no images
        [audios],
        [audio_parts],
        return_tensors="pt",
        max_length=4096,
    )
    return inputs


def main():
    # Load labels
    with open(LABELS_FILE) as f:
        labels = json.load(f)
    logger.info(f"Loaded {len(labels)} labeled samples")

    if len(labels) == 0:
        logger.error("No labels found. Label some samples first with label_audio.py")
        return

    # Load model
    logger.info("Loading model...")
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=False,
        init_audio=True,
        init_tts=False,
    )
    model.prepare_processor(processor=None, tokenizer=None)
    model.processor.tokenizer.padding_side = "right"

    # Apply LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=r"llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    if not hasattr(model, "get_input_embeddings"):
        from types import MethodType
        model.get_input_embeddings = MethodType(
            lambda self: self.llm.get_input_embeddings(), model
        )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.cuda()
    model.train()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # Precompute: tokenize the assistant answer for each sample to know which tokens to supervise
    # We tokenize the target alone to find its length, then mask everything else as -100
    logger.info("Tokenizing samples...")
    samples = []
    for label in labels:
        audio_path = label["audio_path"]
        if not os.path.exists(audio_path):
            logger.warning(f"Audio not found: {audio_path}, skipping")
            continue

        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        target = json.dumps({
            "naturalness": label["naturalness"],
            "noisiness": label["noisiness"],
            "loudness": label["loudness"],
        })

        try:
            inputs = build_tokenized_sample(model, audio, target)
            # Figure out how many tokens the target occupies
            target_tokens = model.processor.tokenizer.encode(target, add_special_tokens=False)
            samples.append((inputs, len(target_tokens)))
            logger.info(f"  Tokenized {label['sample_id']}: {inputs['input_ids'].shape[1]} tokens, target={len(target_tokens)} tokens")
        except Exception as e:
            logger.warning(f"Failed to tokenize {label['sample_id']}: {e}")
            continue

    logger.info(f"Successfully tokenized {len(samples)} samples")
    if len(samples) == 0:
        return

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=0.01,
    )

    # Split into train/val (90/10)
    indices = np.random.permutation(len(samples))
    split = int(len(samples) * TRAIN_RATIO)
    train_indices = indices[:split]
    val_indices = indices[split:]
    logger.info(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

    def compute_loss_on(subset_indices):
        """Compute average loss over a subset without gradient updates."""
        total = 0.0
        for idx in subset_indices:
            inputs, target_len = samples[idx]
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            batch.pop("image_sizes", None)
            input_ids = batch["input_ids"]
            seq_len = input_ids.shape[1]
            labels_tensor = torch.full_like(input_ids, -100, dtype=torch.long)
            target_start = max(0, seq_len - target_len - 5)
            labels_tensor[:, target_start:] = input_ids[:, target_start:].long()
            with torch.no_grad():
                outputs = model.base_model.model.llm(
                    input_ids=input_ids,
                    attention_mask=batch.get("attention_mask"),
                    labels=labels_tensor,
                )
            total += outputs.loss.item()
            del outputs, batch, input_ids, labels_tensor
            torch.cuda.empty_cache()
        return total / max(len(subset_indices), 1)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience_counter = 0

    # Training loop with early stopping on validation loss
    for epoch in range(MAX_EPOCHS):
        model.train()
        total_loss = 0.0
        perm = np.random.permutation(len(train_indices))
        optimizer.zero_grad()

        for step, pi in enumerate(perm):
            idx = train_indices[pi]
            inputs, target_len = samples[idx]

            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            batch.pop("image_sizes", None)

            input_ids = batch["input_ids"]
            seq_len = input_ids.shape[1]

            labels_tensor = torch.full_like(input_ids, -100, dtype=torch.long)
            target_start = max(0, seq_len - target_len - 5)
            labels_tensor[:, target_start:] = input_ids[:, target_start:].long()

            outputs = model.base_model.model.llm(
                input_ids=input_ids,
                attention_mask=batch.get("attention_mask"),
                labels=labels_tensor,
            )
            loss = outputs.loss / GRAD_ACCUM

            loss.backward()
            total_loss += loss.item() * GRAD_ACCUM
            del outputs, loss, batch, input_ids, labels_tensor
            torch.cuda.empty_cache()

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        # Flush remaining gradients
        if len(train_indices) % GRAD_ACCUM != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = total_loss / len(train_indices)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        avg_val_loss = compute_loss_on(val_indices) if len(val_indices) > 0 else avg_train_loss
        val_losses.append(avg_val_loss)

        logger.info(f"Epoch {epoch+1}/{MAX_EPOCHS} - train loss: {avg_train_loss:.4f}, val loss: {avg_val_loss:.4f}")

        # Early stopping (lower loss = better accuracy)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best checkpoint
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            model.save_pretrained(OUTPUT_DIR)
            model.base_model.model.processor.tokenizer.save_pretrained(OUTPUT_DIR)
            logger.info("  Saved best model checkpoint")
        else:
            patience_counter += 1
            logger.info(f"  No improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                logger.info("Early stopping triggered")
                break

        # Plot losses so far
        plt.figure()
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.tight_layout()
        os.makedirs(os.path.dirname(LOSS_PLOT_PATH), exist_ok=True)
        plt.savefig(LOSS_PLOT_PATH)
        plt.close()

    logger.info(f"LoRA adapter saved to {OUTPUT_DIR}")
    logger.info(f"Loss plot saved to {LOSS_PLOT_PATH}")


if __name__ == "__main__":
    main()
