#!/usr/bin/env python3
# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT
"""
Fine-tune Google Gemma 4 (audio-capable) with LoRA + a numeric REGRESSION HEAD
on labeled audio. This is the Gemma counterpart of finetune_audio_scorer_reg.py
(which targets MiniCPM-o 4.5); it keeps the same objective, data split, loss,
early-stopping criterion, and report schema so results are directly comparable.

Design (mirrors the MiniCPM regression trainer):

  1. AUDIO IS ACTUALLY USED. Gemma 4 is natively multimodal: the processor turns
     an audio clip into `input_features` (+ `input_features_mask`) and inserts
     audio soft-tokens into the sequence, so the backbone hidden states are
     audio-conditioned. We run the full ForConditionalGeneration forward with
     output_hidden_states=True and read the last layer's final-token state.

  2. ORDINAL OBJECTIVE. A small linear head maps that final-token hidden state to
     3 continuous scores (naturalness, noisiness, loudness), trained with Huber
     loss on scores normalized to [0,1]. Head trains at 10x the LoRA LR. Early
     stopping maximizes dev mean |Pearson r| (the eval metric).

Reads labels from: data/training_dataset/labels/labels.json
Saves LoRA adapter + regression head + test_correlation.json under OUTPUT_DIR.

Model note: Gemma 4 supports audio on the E2B, E4B, and 12B variants. The 12B
model needs substantial GPU memory even with LoRA + bf16; set AUDIO_SCORER_MODEL
to a smaller audio-capable variant (e.g. google/gemma-4-E4B-it) to fit smaller
GPUs. Verify the exact checkpoint id on Hugging Face before running.

Env-var interface mirrors the MiniCPM trainer: AUDIO_SCORER_MODEL, _OUTPUT_DIR,
_LABELS_FILE, _LR, _HEAD_LR, _LORA_R, _LORA_ALPHA, _LORA_DROPOUT, _LORA_TARGETS,
_PATIENCE, _WEIGHT_DECAY, _MAX_EPOCHS, _GRAD_ACCUM, _SEED.
"""

import json
import logging
import os

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LABELS_FILE = os.environ.get(
    "AUDIO_SCORER_LABELS_FILE", "data/training_dataset/labels/labels.json"
)
OUTPUT_DIR = os.environ.get(
    "AUDIO_SCORER_OUTPUT_DIR", "data/trained_models/gemma4_audio_scorer_reg"
)
# Audio-capable Gemma 4 checkpoint. E4B fits a single mid-size GPU far more
# comfortably than 12B. Confirm the exact id on Hugging Face before running.
MODEL_NAME = os.environ.get("AUDIO_SCORER_MODEL", "google/gemma-4-E4B-it")

SCORING_PROMPT = (
    "You are an expert audio quality evaluator. Listen to this audio carefully and rate it on three dimensions:\n"
    "1. Naturalness (1-10): How natural and human-like does the speech sound?\n"
    "2. Noisiness (1-10): How noisy is the audio? 1=clean, 10=very noisy.\n"
    "3. Loudness (1-10): How loud is the audio? 1=barely audible, 10=very loud."
)

MAX_EPOCHS = int(os.environ.get("AUDIO_SCORER_MAX_EPOCHS", "20"))
LR = float(os.environ.get("AUDIO_SCORER_LR", "3e-4"))
HEAD_LR = float(os.environ.get("AUDIO_SCORER_HEAD_LR", str(LR * 10)))
GRAD_ACCUM = int(os.environ.get("AUDIO_SCORER_GRAD_ACCUM", "4"))
PATIENCE = int(os.environ.get("AUDIO_SCORER_PATIENCE", "5"))
WEIGHT_DECAY = float(os.environ.get("AUDIO_SCORER_WEIGHT_DECAY", "0.01"))
LORA_R = int(os.environ.get("AUDIO_SCORER_LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("AUDIO_SCORER_LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.environ.get("AUDIO_SCORER_LORA_DROPOUT", "0.05"))
LORA_TARGETS = os.environ.get("AUDIO_SCORER_LORA_TARGETS", "attn")
SEED = int(os.environ.get("AUDIO_SCORER_SEED", "42"))
TRAIN_RATIO = 0.70
DEV_RATIO = 0.15
# remaining 0.15 is the held-out test set

LOSS_PLOT_PATH = os.path.join(OUTPUT_DIR, "training_loss.png")
TEST_REPORT_PATH = os.path.join(OUTPUT_DIR, "test_correlation.json")
HEAD_PATH = os.path.join(OUTPUT_DIR, "reg_head.pt")
METRIC_NAMES = ["naturalness", "noisiness", "loudness"]
SCORE_MIN, SCORE_MAX = 1.0, 10.0
AUDIO_SR = 16000  # Gemma3n/4 audio feature extractor expects 16 kHz


def _fmt(x):
    return "nan" if x is None else f"{x:.4f}"


def _norm(v):
    """1-10 -> [0,1]."""
    return (float(v) - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)


def _denorm(x):
    """[0,1] -> 1-10 (clamped)."""
    v = x * (SCORE_MAX - SCORE_MIN) + SCORE_MIN
    return float(np.clip(v, SCORE_MIN, SCORE_MAX))


def build_prompt_inputs(processor, audio_array):
    """Build Gemma processor inputs for prompt + one audio clip (no target).

    Uses apply_chat_template with add_generation_prompt=True so the final
    position is where the model would begin its answer. Returns a dict of
    tensors including input_ids, attention_mask, input_features, and
    input_features_mask.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_array},
                {"type": "text", "text": SCORING_PROMPT},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    return inputs


def _last_token_hidden(model, inputs, reg_head):
    """Full multimodal forward; return head(final-real-token hidden state).

    Returns a (3,) tensor of predicted normalized scores.
    """
    data = {k: (v.cuda() if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()}
    attn = data.get("attention_mask")
    outputs = model(
        **data,
        output_hidden_states=True,
        use_cache=False,
    )
    hs = outputs.hidden_states[-1]  # (1, seq, hidden)
    if attn is not None:
        last_idx = int(attn[0].sum().item()) - 1
    else:
        last_idx = hs.shape[1] - 1
    pooled = hs[0, last_idx, :]
    pred = reg_head(pooled.to(reg_head.weight.dtype))  # (3,)
    return pred


def evaluate_regression(model, reg_head, samples, indices):
    """Predict scores for `indices` and correlate with human labels.

    Same report schema as the MiniCPM trainer so downstream summary tooling
    consumes it unchanged.
    """
    human = {m: [] for m in METRIC_NAMES}
    pred = {m: [] for m in METRIC_NAMES}
    pairs = []

    model.eval()
    reg_head.eval()
    with torch.no_grad():
        for idx in indices:
            sample = samples[idx]
            label = sample["label"]
            out = _last_token_hidden(model, sample["inputs"], reg_head)
            scores = {m: _denorm(float(out[i].item()))
                      for i, m in enumerate(METRIC_NAMES)}
            torch.cuda.empty_cache()
            row = {"sample_id": label.get("sample_id"), "split": label.get("split")}
            for m in METRIC_NAMES:
                human[m].append(float(label[m]))
                pred[m].append(scores[m])
                row[f"human_{m}"] = float(label[m])
                row[f"pred_{m}"] = scores[m]
            pairs.append(row)

    def _corr(h, p):
        h = np.asarray(h, dtype=float)
        p = np.asarray(p, dtype=float)
        n = len(h)
        result = {
            "n": int(n),
            "pearson_r": None, "pearson_p": None,
            "spearman_rho": None, "spearman_p": None,
            "mae": None,
        }
        if n >= 2:
            result["mae"] = float(np.mean(np.abs(h - p)))
            if np.std(h) > 0 and np.std(p) > 0:
                pr = pearsonr(h, p)
                sr = spearmanr(h, p)
                result["pearson_r"] = float(pr.statistic)
                result["pearson_p"] = float(pr.pvalue)
                result["spearman_rho"] = float(sr.statistic)
                result["spearman_p"] = float(sr.pvalue)
        return result

    metrics = {m: _corr(human[m], pred[m]) for m in METRIC_NAMES}
    rs = [abs(metrics[m]["pearson_r"]) for m in METRIC_NAMES
          if metrics[m]["pearson_r"] is not None]
    mean_abs_r = float(np.mean(rs)) if rs else 0.0
    return {
        "seed": SEED,
        "objective": "regression_head",
        "model": MODEL_NAME,
        "split_ratios": {"train": TRAIN_RATIO, "dev": DEV_RATIO,
                         "test": round(1 - TRAIN_RATIO - DEV_RATIO, 4)},
        "n_test_requested": int(len(indices)),
        "n_test_scored": len(pairs),
        "parse_failures": 0,
        "mean_abs_pearson_r": mean_abs_r,
        "metrics": metrics,
        "pairs": pairs,
    }


# LoRA target-module presets for Gemma's language backbone. Gemma names its
# attention submodules q_proj/k_proj/v_proj/o_proj and MLP gate/up/down_proj,
# under the language_model.layers.* path inside ForConditionalGeneration.
_TARGET_PRESETS = {
    "attn": r"language_model\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)",
    "attn_mlp": r"language_model\..*layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))",
}


def main():
    with open(LABELS_FILE) as f:
        labels = json.load(f)
    logger.info(f"Loaded {len(labels)} labeled samples")
    if len(labels) == 0:
        logger.error("No labels found.")
        return

    logger.info(f"Loading processor + model: {MODEL_NAME}")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    target_modules = _TARGET_PRESETS.get(LORA_TARGETS, _TARGET_PRESETS["attn"])
    logger.info(
        f"[gemma reg] LoRA r={LORA_R} alpha={LORA_ALPHA} dropout={LORA_DROPOUT} "
        f"targets={LORA_TARGETS} | LR={LR} head_lr={HEAD_LR} "
        f"weight_decay={WEIGHT_DECAY} patience={PATIENCE} "
        f"max_epochs={MAX_EPOCHS} seed={SEED}"
    )
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=target_modules,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.cuda()

    # Hidden size of the text backbone (Gemma4 config nests it under text_config).
    cfg = model.base_model.model.config
    hidden_size = getattr(cfg, "text_config", cfg).hidden_size
    reg_head = nn.Linear(hidden_size, len(METRIC_NAMES)).cuda().to(torch.bfloat16)

    logger.info("Preparing samples (prompt + audio)...")
    samples = []
    for label in labels:
        audio_path = label["audio_path"]
        if not os.path.exists(audio_path):
            logger.warning(f"Audio not found: {audio_path}, skipping")
            continue
        audio, _ = librosa.load(audio_path, sr=AUDIO_SR, mono=True)
        target = torch.tensor(
            [_norm(label[m]) for m in METRIC_NAMES], dtype=torch.float32
        )
        try:
            inputs = build_prompt_inputs(processor, audio)
            samples.append({"inputs": inputs, "target": target, "label": label})
        except Exception as e:
            logger.warning(f"Failed to prepare {label.get('sample_id')}: {e}")
            continue
    logger.info(f"Prepared {len(samples)} samples")
    if len(samples) == 0:
        return

    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": LR, "weight_decay": WEIGHT_DECAY},
            {"params": reg_head.parameters(), "lr": HEAD_LR, "weight_decay": WEIGHT_DECAY},
        ]
    )
    loss_fn = nn.HuberLoss(delta=0.1)

    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(samples))
    n = len(samples)
    train_split = int(n * TRAIN_RATIO)
    dev_split = int(n * (TRAIN_RATIO + DEV_RATIO))
    train_indices = indices[:train_split]
    dev_indices = indices[train_split:dev_split]
    test_indices = indices[dev_split:]
    logger.info(
        f"Split (seed={SEED}) -> train: {len(train_indices)}, "
        f"dev: {len(dev_indices)}, test (held-out): {len(test_indices)}"
    )

    train_losses, dev_scores = [], []
    best_metric = -float("inf")
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        model.train(); reg_head.train()
        total_loss = 0.0
        perm = rng.permutation(len(train_indices))
        optimizer.zero_grad()

        for step, pi in enumerate(perm):
            idx = train_indices[pi]
            s = samples[idx]
            pred = _last_token_hidden(model, s["inputs"], reg_head)
            loss = loss_fn(pred.float(), s["target"].cuda()) / GRAD_ACCUM
            loss.backward()
            total_loss += loss.item() * GRAD_ACCUM
            torch.cuda.empty_cache()

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(lora_params + list(reg_head.parameters()), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        if len(train_indices) % GRAD_ACCUM != 0:
            torch.nn.utils.clip_grad_norm_(lora_params + list(reg_head.parameters()), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = total_loss / len(train_indices)
        train_losses.append(avg_train_loss)

        dev_report = evaluate_regression(model, reg_head, samples, dev_indices) \
            if len(dev_indices) > 0 else None
        dev_mean_r = dev_report["mean_abs_pearson_r"] if dev_report else 0.0
        dev_scores.append(dev_mean_r)
        logger.info(
            f"Epoch {epoch+1}/{MAX_EPOCHS} - train Huber: {avg_train_loss:.4f}, "
            f"dev mean|r|: {dev_mean_r:.4f}"
        )

        if dev_mean_r > best_metric:
            best_metric = dev_mean_r
            patience_counter = 0
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            model.save_pretrained(OUTPUT_DIR)
            processor.save_pretrained(OUTPUT_DIR)
            torch.save(reg_head.state_dict(), HEAD_PATH)
            logger.info("  Saved best checkpoint (LoRA + reg head)")
        else:
            patience_counter += 1
            logger.info(f"  No improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                logger.info("Early stopping triggered")
                break

        plt.figure()
        ax1 = plt.gca()
        ax1.plot(range(1, len(train_losses) + 1), train_losses, "b-", label="Train Huber")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train Huber loss", color="b")
        ax2 = ax1.twinx()
        ax2.plot(range(1, len(dev_scores) + 1), dev_scores, "g-", label="Dev mean|r|")
        ax2.set_ylabel("Dev mean|r|", color="g")
        plt.title(f"Gemma regression trainer ({MODEL_NAME})")
        plt.tight_layout()
        os.makedirs(os.path.dirname(LOSS_PLOT_PATH), exist_ok=True)
        plt.savefig(LOSS_PLOT_PATH)
        plt.close()

    logger.info(f"Best dev mean|r|: {best_metric:.4f}")

    if len(test_indices) == 0:
        logger.warning("No held-out test samples; skipping correlation evaluation.")
        return

    if os.path.exists(HEAD_PATH):
        reg_head.load_state_dict(torch.load(HEAD_PATH))

    logger.info(f"Evaluating on {len(test_indices)} held-out test samples...")
    report = evaluate_regression(model, reg_head, samples, test_indices)
    os.makedirs(os.path.dirname(TEST_REPORT_PATH), exist_ok=True)
    with open(TEST_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Test-set correlation report saved to {TEST_REPORT_PATH}")

    logger.info("Held-out test correlation (Gemma regression model vs human):")
    for metric in METRIC_NAMES:
        m = report["metrics"][metric]
        logger.info(
            f"  {metric:12s} n={m['n']:3d}  "
            f"Pearson r={_fmt(m['pearson_r'])} (p={_fmt(m['pearson_p'])})  "
            f"Spearman rho={_fmt(m['spearman_rho'])}  MAE={_fmt(m['mae'])}"
        )
    logger.info(f"  mean|r| = {report['mean_abs_pearson_r']:.4f}")


if __name__ == "__main__":
    main()
