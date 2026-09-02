#!/usr/bin/env python3
# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT
"""
Fine-tune MiniCPM-o 4.5 with LoRA + a numeric REGRESSION HEAD on labeled audio.

Two important design choices:

  1. AUDIO IS ACTUALLY USED. We call the base MiniCPMO multimodal forward
     `model.base_model.model(data, output_hidden_states=True)`, which runs
     get_vllm_embedding + get_omni_embedding to splice audio embeddings into the
     sequence before the LLM, so the hidden states are audio-conditioned.

  2. ORDINAL OBJECTIVE. A small linear head maps the final-token hidden state to
     3 continuous scores (naturalness, noisiness, loudness), trained with Huber
     loss on scores normalized to [0,1]. This aligns the training objective with
     the evaluation metric (Pearson/Spearman correlation vs human labels).

Reads labels from: data/training_dataset/labels/labels.json
Saves LoRA adapter + regression head + test_correlation.json under OUTPUT_DIR.

Env-var interface: AUDIO_SCORER_OUTPUT_DIR, _LR, _LORA_R, _LORA_ALPHA,
_LORA_DROPOUT, _LORA_TARGETS, _PATIENCE, _WEIGHT_DECAY, _MAX_EPOCHS, _GRAD_ACCUM,
_SEED.
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
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModel
from peft import LoraConfig, get_peft_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LABELS_FILE = os.environ.get(
    "AUDIO_SCORER_LABELS_FILE", "data/training_dataset/labels/labels.json"
)
OUTPUT_DIR = os.environ.get(
    "AUDIO_SCORER_OUTPUT_DIR", "data/trained_models/minicpm_audio_scorer_reg"
)
MODEL_NAME = "openbmb/MiniCPM-o-4_5"

# Prompt is the user turn only; we regress from the hidden state at the final
# prompt token (with add_generation_prompt=True). No assistant/JSON target.
SCORING_PROMPT = (
    "You are an expert audio quality evaluator. Listen to this audio carefully and rate it on three dimensions:\n"
    "1. Naturalness (1-10): How natural and human-like does the speech sound?\n"
    "2. Noisiness (1-10): How noisy is the audio? 1=clean, 10=very noisy.\n"
    "3. Loudness (1-10): How loud is the audio? 1=barely audible, 10=very loud."
)

# Defaults below are the best config from the regression-head sweep ("lr_hi":
# held-out mean|Pearson r| = 0.679 on the 404-sample / n=61 test split).
# All values remain env-overridable so the sweep harness can vary them.
MAX_EPOCHS = int(os.environ.get("AUDIO_SCORER_MAX_EPOCHS", "20"))
LR = float(os.environ.get("AUDIO_SCORER_LR", "3e-4"))
# Head trains faster than the LoRA backbone. Best config used an explicit 3e-3
# (10x LR); if LR is overridden without HEAD_LR, fall back to 10x that LR.
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

# Scores are on a 1-10 integer scale; normalize to [0,1] for a well-conditioned
# regression target, then map predictions back to 1-10 for reporting.
SCORE_MIN, SCORE_MAX = 1.0, 10.0


def _fmt(x):
    return "nan" if x is None else f"{x:.4f}"


def _norm(v):
    """1-10 -> [0,1]."""
    return (float(v) - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)


def _denorm(x):
    """[0,1] -> 1-10 (clamped)."""
    v = x * (SCORE_MAX - SCORE_MIN) + SCORE_MIN
    return float(np.clip(v, SCORE_MIN, SCORE_MAX))


def build_prompt_inputs(model, audio_array):
    """Tokenize prompt + audio into processor inputs (no assistant target).

    Mirrors the message-flattening that chat() does, but stops at the generation
    prompt so the final position is where the model would begin its answer.
    """
    msgs = [{"role": "user", "content": [SCORING_PROMPT, audio_array]}]
    copy_msgs = deepcopy(msgs)
    audios = []
    audio_parts = []
    for i, msg in enumerate(copy_msgs):
        cur_msgs = []
        for c in msg["content"]:
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
        add_generation_prompt=True,
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


def _last_token_hidden(model, inputs, reg_head):
    """Run the full multimodal forward and return head(last_real_token_hidden).

    Calls the base MiniCPMO forward (model.base_model.model) with the processor
    dict as the positional `data` arg so audio embeddings are fused into the
    sequence (get_vllm_embedding + get_omni_embedding) before the LLM. Calling
    the PeftModel wrapper directly would unpack the dict into kwargs and lose
    `data`; LoRA submodules are still active on the base model.

    Returns a (3,) tensor of predicted normalized scores.
    """
    data = {k: (v.cuda() if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()}
    data.pop("image_sizes", None)
    # Skip the training-mode dummy-vision path (no vpm/resampler under
    # init_vision=False): present key -> returned as-is. Empty == no vision.
    data["vision_hidden_states"] = [[]]
    input_ids = data["input_ids"]
    # MiniCPMO.forward requires position_ids; the processor doesn't emit them.
    attn = data.get("attention_mask")
    if attn is None:
        attn = torch.ones_like(input_ids)
    data["position_ids"] = (attn.long().cumsum(-1) - 1).clamp(min=0)
    outputs = model.base_model.model(
        data, output_hidden_states=True, use_cache=False
    )
    hs = outputs.hidden_states[-1]  # (1, seq, hidden)

    attn = inputs.get("attention_mask")
    if attn is not None:
        last_idx = int(attn[0].sum().item()) - 1
    else:
        last_idx = hs.shape[1] - 1
    pooled = hs[0, last_idx, :]  # (hidden,)
    pred = reg_head(pooled.to(reg_head.weight.dtype))  # (3,)
    return pred


def evaluate_regression(model, reg_head, samples, indices):
    """Predict scores for `indices` and correlate with human labels.

    Returns the same report schema as the generative trainer's
    evaluate_test_correlation so summarize_sweep.py can consume it unchanged.
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
        "split_ratios": {"train": TRAIN_RATIO, "dev": DEV_RATIO,
                         "test": round(1 - TRAIN_RATIO - DEV_RATIO, 4)},
        "n_test_requested": int(len(indices)),
        "n_test_scored": len(pairs),
        "parse_failures": 0,
        "mean_abs_pearson_r": mean_abs_r,
        "metrics": metrics,
        "pairs": pairs,
    }


def main():
    with open(LABELS_FILE) as f:
        labels = json.load(f)
    logger.info(f"Loaded {len(labels)} labeled samples")
    if len(labels) == 0:
        logger.error("No labels found.")
        return

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

    _TARGET_PRESETS = {
        "attn": r"llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)",
        "attn_mlp": r"llm\..*layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))",
    }
    target_modules = _TARGET_PRESETS.get(LORA_TARGETS, _TARGET_PRESETS["attn"])
    logger.info(
        f"[regression] LoRA r={LORA_R} alpha={LORA_ALPHA} dropout={LORA_DROPOUT} "
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

    if not hasattr(model, "get_input_embeddings"):
        from types import MethodType
        model.get_input_embeddings = MethodType(
            lambda self: self.llm.get_input_embeddings(), model
        )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.cuda()
    # NOTE: neither gradient_checkpointing_enable() nor enable_input_require_grads()
    # are used. The multimodal forward splices audio embeddings into inputs_embeds
    # via an in-place write (get_omni_embedding: input_embeddings[i, bound] = ...).
    # Making the embedding output a grad-requiring leaf (enable_input_require_grads)
    # or wrapping it in gradient checkpointing makes that in-place write raise
    # "a view of a leaf Variable ... in-place operation". LoRA + head gradients
    # still flow without either call. Tradeoff: higher activation memory (fine at
    # batch size 1).

    hidden_size = model.base_model.model.llm.config.hidden_size
    reg_head = nn.Linear(hidden_size, len(METRIC_NAMES)).cuda().to(torch.bfloat16)

    # Tokenize/prepare every sample once (prompt + audio -> processor inputs).
    logger.info("Preparing samples (prompt + audio)...")
    samples = []
    for label in labels:
        audio_path = label["audio_path"]
        if not os.path.exists(audio_path):
            logger.warning(f"Audio not found: {audio_path}, skipping")
            continue
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        target = torch.tensor(
            [_norm(label[m]) for m in METRIC_NAMES], dtype=torch.float32
        )
        try:
            inputs = build_prompt_inputs(model, audio)
            samples.append({"inputs": inputs, "target": target, "label": label})
        except Exception as e:
            logger.warning(f"Failed to prepare {label.get('sample_id')}: {e}")
            continue
    logger.info(f"Prepared {len(samples)} samples")
    if len(samples) == 0:
        return

    # AdamW over LoRA params (LR) + head params (HEAD_LR).
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

    def dev_loss_on(subset):
        model.eval(); reg_head.eval()
        total = 0.0
        with torch.no_grad():
            for idx in subset:
                s = samples[idx]
                pred = _last_token_hidden(model, s["inputs"], reg_head)
                loss = loss_fn(pred.float(), s["target"].cuda())
                total += loss.item()
                torch.cuda.empty_cache()
        return total / max(len(subset), 1)

    train_losses, dev_scores = [], []
    best_metric = -float("inf")  # maximize dev mean|r|
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

        # Early stopping on dev mean|Pearson r| (aligned with the eval metric).
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
            model.base_model.model.processor.tokenizer.save_pretrained(OUTPUT_DIR)
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
        plt.title("Regression trainer: loss & dev correlation")
        plt.tight_layout()
        os.makedirs(os.path.dirname(LOSS_PLOT_PATH), exist_ok=True)
        plt.savefig(LOSS_PLOT_PATH)
        plt.close()

    logger.info(f"Best dev mean|r|: {best_metric:.4f}")

    if len(test_indices) == 0:
        logger.warning("No held-out test samples; skipping correlation evaluation.")
        return

    # Reload best checkpoint's head for the final held-out evaluation.
    if os.path.exists(HEAD_PATH):
        reg_head.load_state_dict(torch.load(HEAD_PATH))

    logger.info(f"Evaluating on {len(test_indices)} held-out test samples...")
    report = evaluate_regression(model, reg_head, samples, test_indices)
    os.makedirs(os.path.dirname(TEST_REPORT_PATH), exist_ok=True)
    with open(TEST_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Test-set correlation report saved to {TEST_REPORT_PATH}")

    logger.info("Held-out test correlation (regression model vs human):")
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
