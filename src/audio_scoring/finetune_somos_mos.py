#!/usr/bin/env python3
# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT
"""Fine-tune MiniCPM-o 4.5 with LoRA + a numeric REGRESSION HEAD on SOMOS MOS.

Adapted from finetune_audio_scorer_reg.py. Key differences:

  1. SINGLE METRIC: predicts one value, MOS (Mean Opinion Score), instead of the
     three naturalness/noisiness/loudness dimensions.

  2. 1-5 SCALE: SOMOS MOS labels are human opinion scores on a 1-5 scale (not the
     1-10 naturalness scale of the original trainer). SCORE_MIN/MAX = 1.0/5.0.

  3. OFFICIAL SPLITS: uses SOMOS's own train/valid/test assignment from the
     labels file's "split" field, rather than a random 70/15/15 re-split, so we
     respect SOMOS's system/speaker-leakage-safe partitioning.

  4. MOS PROMPT: asks the model for an overall 1-5 MOS rating.

Everything else (audio-fused multimodal forward, last-token pooling, Huber loss
on [0,1]-normalized targets, LoRA config, early stopping on dev correlation) is
unchanged from the regression trainer.

Reads labels from: data/training_dataset/labels/somos_mos_labels.json
Saves LoRA adapter + reg head + test_correlation.json under OUTPUT_DIR.
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
    "SOMOS_LABELS_FILE", "data/training_dataset/labels/somos_mos_labels.json"
)
OUTPUT_DIR = os.environ.get(
    "SOMOS_MOS_OUTPUT_DIR", "data/trained_models/minicpm_somos_mos_reg"
)
MODEL_NAME = "openbmb/MiniCPM-o-4_5"

# Single-metric MOS prompt on a 1-5 scale.
SCORING_PROMPT = (
    "You are an expert audio quality evaluator. Listen to this audio carefully "
    "and rate its overall quality using the Mean Opinion Score (MOS) scale from "
    "1 to 5, where 1 is bad, 2 is poor, 3 is fair, 4 is good, and 5 is excellent."
)

MAX_EPOCHS = int(os.environ.get("SOMOS_MOS_MAX_EPOCHS", "20"))
LR = float(os.environ.get("SOMOS_MOS_LR", "3e-4"))
HEAD_LR = float(os.environ.get("SOMOS_MOS_HEAD_LR", str(LR * 10)))
GRAD_ACCUM = int(os.environ.get("SOMOS_MOS_GRAD_ACCUM", "4"))
PATIENCE = int(os.environ.get("SOMOS_MOS_PATIENCE", "5"))
WEIGHT_DECAY = float(os.environ.get("SOMOS_MOS_WEIGHT_DECAY", "0.01"))
LORA_R = int(os.environ.get("SOMOS_MOS_LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("SOMOS_MOS_LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.environ.get("SOMOS_MOS_LORA_DROPOUT", "0.05"))
LORA_TARGETS = os.environ.get("SOMOS_MOS_LORA_TARGETS", "attn")
SEED = int(os.environ.get("SOMOS_MOS_SEED", "42"))

LOSS_PLOT_PATH = os.path.join(OUTPUT_DIR, "training_loss.png")
TEST_REPORT_PATH = os.path.join(OUTPUT_DIR, "test_correlation.json")
HEAD_PATH = os.path.join(OUTPUT_DIR, "reg_head.pt")
# Resumable training state. BEST_DIR holds the best-by-dev checkpoint (what you
# deploy); CKPT_DIR holds the latest end-of-epoch state for stop/resume.
BEST_DIR = os.path.join(OUTPUT_DIR, "best")
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoint_last")
TRAIN_STATE_PATH = os.path.join(CKPT_DIR, "train_state.pt")
# Set SOMOS_MOS_RESUME=0 to force a fresh start even if a checkpoint exists.
RESUME = os.environ.get("SOMOS_MOS_RESUME", "1") not in ("0", "false", "False", "")
METRIC_NAMES = ["mos"]

# MOS is on a 1-5 scale; normalize to [0,1] for a well-conditioned target.
SCORE_MIN, SCORE_MAX = 1.0, 5.0


def _fmt(x):
    return "nan" if x is None else f"{x:.4f}"


def _norm(v):
    """1-5 -> [0,1]."""
    return (float(v) - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)


def _denorm(x):
    """[0,1] -> 1-5 (clamped)."""
    v = x * (SCORE_MAX - SCORE_MIN) + SCORE_MIN
    return float(np.clip(v, SCORE_MIN, SCORE_MAX))


def build_prompt_inputs(model, audio_array):
    """Tokenize prompt + audio into processor inputs (no assistant target)."""
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
    """Run the full multimodal forward; return head(last_real_token_hidden)."""
    data = {k: (v.cuda() if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()}
    data.pop("image_sizes", None)
    data["vision_hidden_states"] = [[]]
    input_ids = data["input_ids"]
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
    pred = reg_head(pooled.to(reg_head.weight.dtype))  # (len(METRIC_NAMES),)
    return pred


def evaluate_regression(model, reg_head, samples, indices):
    """Predict MOS for `indices` and correlate with human labels."""
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
        "scale": {"min": SCORE_MIN, "max": SCORE_MAX},
        "split_source": "somos_official",
        "n_test_requested": int(len(indices)),
        "n_test_scored": len(pairs),
        "parse_failures": 0,
        "mean_abs_pearson_r": mean_abs_r,
        "metrics": metrics,
        "pairs": pairs,
    }


def save_checkpoint(model, reg_head, optimizer, scheduler, epoch,
                    best_metric, patience_counter, train_losses, dev_scores,
                    numpy_rng, torch_rng_state):
    """Atomically persist full training state for stop/resume.

    Saves everything needed to continue training byte-for-byte:
      - LoRA adapter + reg head weights (resumable model state)
      - optimizer state (AdamW moments) and LR scheduler state
      - epoch counter, early-stopping bookkeeping, loss/score history
      - numpy + torch RNG states (so the sample shuffle order is reproducible)
    """
    os.makedirs(CKPT_DIR, exist_ok=True)
    # LoRA adapter + head into the checkpoint dir.
    model.save_pretrained(CKPT_DIR)
    model.base_model.model.processor.tokenizer.save_pretrained(CKPT_DIR)
    torch.save(reg_head.state_dict(), os.path.join(CKPT_DIR, "reg_head.pt"))

    state = {
        "epoch": epoch,  # number of COMPLETED epochs
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_metric": best_metric,
        "patience_counter": patience_counter,
        "train_losses": train_losses,
        "dev_scores": dev_scores,
        "numpy_rng": numpy_rng.bit_generator.state,
        "torch_rng": torch_rng_state,
        "cuda_rng": (torch.cuda.get_rng_state_all()
                     if torch.cuda.is_available() else None),
        "config": {
            "lr": LR, "head_lr": HEAD_LR, "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
            "lora_targets": LORA_TARGETS, "weight_decay": WEIGHT_DECAY,
            "grad_accum": GRAD_ACCUM, "max_epochs": MAX_EPOCHS, "seed": SEED,
            "score_min": SCORE_MIN, "score_max": SCORE_MAX,
        },
    }
    # Atomic write: temp file then rename, so a crash mid-save can't corrupt it.
    tmp = TRAIN_STATE_PATH + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, TRAIN_STATE_PATH)
    logger.info(f"  [checkpoint] saved resumable state at epoch {epoch} -> {CKPT_DIR}")


def load_checkpoint(model, reg_head, optimizer, scheduler):
    """Restore full training state saved by save_checkpoint.

    Returns a dict with start_epoch / best_metric / patience_counter /
    train_losses / dev_scores / numpy_rng_state, or None if nothing to resume.
    LoRA + head weights are loaded from CKPT_DIR into the already-built modules.
    """
    if not (RESUME and os.path.isfile(TRAIN_STATE_PATH)):
        return None
    logger.info(f"Resuming from checkpoint: {TRAIN_STATE_PATH}")
    state = torch.load(TRAIN_STATE_PATH, map_location="cpu", weights_only=False)

    # Reload LoRA adapter weights into the live PEFT model.
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file as _load_safetensors
    adapter_bin = os.path.join(CKPT_DIR, "adapter_model.safetensors")
    adapter_pt = os.path.join(CKPT_DIR, "adapter_model.bin")
    if os.path.isfile(adapter_bin):
        set_peft_model_state_dict(model, _load_safetensors(adapter_bin))
    elif os.path.isfile(adapter_pt):
        set_peft_model_state_dict(model, torch.load(adapter_pt, map_location="cpu"))
    else:
        logger.warning("  No adapter weights found in checkpoint; keeping current LoRA init.")

    head_path = os.path.join(CKPT_DIR, "reg_head.pt")
    if os.path.isfile(head_path):
        reg_head.load_state_dict(torch.load(head_path, map_location="cuda"))

    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])

    # Restore RNG so the epoch/sample shuffle sequence continues identically.
    if state.get("torch_rng") is not None:
        torch.set_rng_state(
            state["torch_rng"].cpu() if hasattr(state["torch_rng"], "cpu")
            else state["torch_rng"]
        )
    if state.get("cuda_rng") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
        except Exception as e:
            logger.warning(f"  Could not restore CUDA RNG: {e}")

    logger.info(
        f"  Resumed at epoch {state['epoch']} "
        f"(best dev mean|r| so far: {state['best_metric']:.4f}, "
        f"patience {state['patience_counter']}/{PATIENCE})"
    )
    return {
        "start_epoch": state["epoch"],
        "best_metric": state["best_metric"],
        "patience_counter": state["patience_counter"],
        "train_losses": state["train_losses"],
        "dev_scores": state["dev_scores"],
        "numpy_rng_state": state["numpy_rng"],
    }


def main():
    with open(LABELS_FILE) as f:
        labels = json.load(f)
    logger.info(f"Loaded {len(labels)} labeled samples from {LABELS_FILE}")
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
        f"[somos-mos] LoRA r={LORA_R} alpha={LORA_ALPHA} dropout={LORA_DROPOUT} "
        f"targets={LORA_TARGETS} | LR={LR} head_lr={HEAD_LR} "
        f"weight_decay={WEIGHT_DECAY} patience={PATIENCE} "
        f"max_epochs={MAX_EPOCHS} seed={SEED} scale=[{SCORE_MIN},{SCORE_MAX}]"
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
    # LR schedule. The ORIGINAL trainer used a flat (constant) LR, so the
    # scheduler is OFF by default to match that approach. Set
    # SOMOS_MOS_COSINE=1 to re-enable cosine decay. When off, a no-op constant
    # schedule keeps the checkpoint/resume plumbing uniform.
    if os.environ.get("SOMOS_MOS_COSINE", "0") in ("1", "true", "True"):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(MAX_EPOCHS, 1)
        )
        logger.info("LR schedule: cosine annealing")
    else:
        # ConstantLR with factor=1.0 => flat LR (matches original trainer),
        # while still exposing .step()/.state_dict() for the resume logic.
        scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=0
        )
        logger.info("LR schedule: flat/constant (matches original approach)")
    loss_fn = nn.HuberLoss(delta=0.1)

    # SOMOS official splits (from the labels "split" field), not a random re-split.
    train_indices = [i for i, s in enumerate(samples) if s["label"].get("split") == "train"]
    dev_indices = [i for i, s in enumerate(samples) if s["label"].get("split") == "valid"]
    test_indices = [i for i, s in enumerate(samples) if s["label"].get("split") == "test"]
    # Fallback: if the labels carry no usable split field, do a seeded 70/15/15.
    if not train_indices:
        logger.warning("No 'train' split entries found; falling back to random 70/15/15.")
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(len(samples))
        n = len(samples)
        a, b = int(n * 0.70), int(n * 0.85)
        train_indices = list(perm[:a]); dev_indices = list(perm[a:b]); test_indices = list(perm[b:])
    logger.info(
        f"SOMOS splits -> train: {len(train_indices)}, "
        f"dev(valid): {len(dev_indices)}, test: {len(test_indices)}"
    )

    rng = np.random.default_rng(SEED)
    train_losses, dev_scores = [], []
    best_metric = -float("inf")  # maximize dev mean|r|
    patience_counter = 0
    start_epoch = 0

    # --- Resume from last checkpoint if present ---------------------------
    resumed = load_checkpoint(model, reg_head, optimizer, scheduler)
    if resumed is not None:
        start_epoch = resumed["start_epoch"]
        best_metric = resumed["best_metric"]
        patience_counter = resumed["patience_counter"]
        train_losses = resumed["train_losses"]
        dev_scores = resumed["dev_scores"]
        # Restore the numpy RNG bit-generator state so the per-epoch train
        # shuffle order continues exactly where it left off.
        rng.bit_generator.state = resumed["numpy_rng_state"]
        if start_epoch >= MAX_EPOCHS:
            logger.info(
                f"Checkpoint already at epoch {start_epoch} >= MAX_EPOCHS "
                f"({MAX_EPOCHS}); nothing left to train. Proceeding to test eval."
            )

    for epoch in range(start_epoch, MAX_EPOCHS):
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
            # Best-by-dev checkpoint -> BEST_DIR (plus legacy top-level copy for
            # compatibility with existing scoring scripts that read OUTPUT_DIR).
            os.makedirs(BEST_DIR, exist_ok=True)
            model.save_pretrained(BEST_DIR)
            model.base_model.model.processor.tokenizer.save_pretrained(BEST_DIR)
            torch.save(reg_head.state_dict(), os.path.join(BEST_DIR, "reg_head.pt"))
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            model.save_pretrained(OUTPUT_DIR)
            model.base_model.model.processor.tokenizer.save_pretrained(OUTPUT_DIR)
            torch.save(reg_head.state_dict(), HEAD_PATH)
            improved = True
            logger.info("  Saved best checkpoint (LoRA + reg head)")
        else:
            patience_counter += 1
            improved = False
            logger.info(f"  No improvement ({patience_counter}/{PATIENCE})")

        # Step the LR schedule once per epoch (state is checkpointed below).
        scheduler.step()

        # Always persist full resumable state at the end of every epoch, so a
        # stop/crash at any point resumes from the last completed epoch with the
        # optimizer, LR schedule, RNG, and early-stopping counters intact.
        save_checkpoint(
            model, reg_head, optimizer, scheduler,
            epoch=epoch + 1,  # completed epochs
            best_metric=best_metric,
            patience_counter=patience_counter,
            train_losses=train_losses,
            dev_scores=dev_scores,
            numpy_rng=rng,
            torch_rng_state=torch.get_rng_state(),
        )

        plt.figure()
        ax1 = plt.gca()
        ax1.plot(range(1, len(train_losses) + 1), train_losses, "b-", label="Train Huber")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train Huber loss", color="b")
        ax2 = ax1.twinx()
        ax2.plot(range(1, len(dev_scores) + 1), dev_scores, "g-", label="Dev mean|r|")
        ax2.set_ylabel("Dev mean|r|", color="g")
        plt.title("SOMOS MOS regression trainer: loss & dev correlation")
        plt.tight_layout()
        os.makedirs(os.path.dirname(LOSS_PLOT_PATH), exist_ok=True)
        plt.savefig(LOSS_PLOT_PATH)
        plt.close()

        if not improved and patience_counter >= PATIENCE:
            logger.info("Early stopping triggered")
            break

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

    logger.info("Held-out test correlation (SOMOS MOS model vs human):")
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
