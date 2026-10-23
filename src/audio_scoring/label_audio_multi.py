#!/usr/bin/env python3
# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT
"""
Per-labeler audio labeling tool for inter-annotator agreement.

This is a labeler-aware variant of label_audio.py. Each labeler's ratings are
written to a SEPARATE directory so the original annotator's labels are never
overwritten. Run one instance per labeler. All three metrics (naturalness,
noisiness, loudness) are collected on one screen per clip.

Usage:
    python src/audio_scoring/label_audio_multi.py --labeler bob [--port 8899]
    # a second labeler, on another port / machine:
    python src/audio_scoring/label_audio_multi.py --labeler carol --port 8900

    # restrict to the held-out test set (the r=0.71 agreement ceiling study):
    python src/audio_scoring/label_audio_multi.py --labeler bob \\
        --only-ids data/training_dataset/labels/test76_ids.json

    # FAST single-metric pass (big 1-10 buttons + number-key shortcuts);
    # naturalness alone is enough for the r=0.71 ceiling:
    python src/audio_scoring/label_audio_multi.py --labeler bob --metric naturalness \\
        --only-ids data/training_dataset/labels/test76_ids.json

Output (one file per metric, same schema as label_audio.py):
    data/training_dataset/labels/by_labeler/<labeler>/naturalness.json
    data/training_dataset/labels/by_labeler/<labeler>/noisiness.json
    data/training_dataset/labels/by_labeler/<labeler>/loudness.json

Progress is saved after every clip, so closing the browser is safe; re-run the
same command to resume. Afterwards, use analyze_agreement.py to compute
inter-annotator agreement (ICC, Krippendorff's alpha) against the original
labels in data/training_dataset/labels/<metric>.json.

Security: the server has no authentication. Binds to 127.0.0.1 by default. To
let a remote labeler use it, prefer an SSH tunnel (ssh -L 8899:localhost:8899
host) over --host 0.0.0.0.
"""

import argparse
import json
import os
import re
import string
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

SAMPLES_DIR = "data/training_dataset"
SAMPLES_JSON = os.path.join(SAMPLES_DIR, "samples.json")
LABELS_JSON = os.path.join(SAMPLES_DIR, "labels", "labels.json")
AUDIO_DIR = os.path.join(SAMPLES_DIR, "audio")
LABELS_DIR = os.path.join(SAMPLES_DIR, "labels")
METRICS = ["naturalness", "noisiness", "loudness"]

METRIC_HINTS = {
    "naturalness": "1 = very robotic/synthetic  |  10 = perfectly human-like",
    "noisiness": "1 = very clean/clear  |  10 = extremely noisy",
    "loudness": "1 = barely audible  |  10 = very loud",
}


def load_samples():
    with open(SAMPLES_JSON) as f:
        return json.load(f)["samples"]


def load_only_ids(path):
    """Read an allow-list of sample_ids from a JSON file.

    Accepts either a bare list of ids, or an object with a 'sample_ids' key
    (the format written by test76_ids.json).
    """
    with open(path) as f:
        data = json.load(f)
    ids = data["sample_ids"] if isinstance(data, dict) else data
    if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
        raise ValueError(f"{path} must contain a list of sample_id strings")
    return list(dict.fromkeys(ids))  # de-dupe, preserve order


def build_samples_for_ids(ids):
    """Build clip entries for an explicit id list.

    Metadata is taken from labels.json (the authoritative 504-sample source,
    a superset of samples.json). Audio is resolved by <id>.wav in AUDIO_DIR,
    because the audio_path recorded in labels.json can be stale. Any id whose
    wav is missing on disk is dropped and reported.
    """
    meta = {l["sample_id"]: l for l in json.load(open(LABELS_JSON))}
    # samples.json adds transcripts for the ids it covers (nice-to-have only).
    try:
        trans = {s["id"]: s.get("transcript", "") for s in load_samples()}
    except Exception:
        trans = {}
    out, missing_meta, missing_audio = [], [], []
    for sid in ids:
        if sid not in meta:
            missing_meta.append(sid)
            continue
        wav = os.path.join(AUDIO_DIR, f"{sid}.wav")
        if not os.path.exists(wav):
            missing_audio.append(sid)
            continue
        out.append({
            "id": sid,
            "split": meta[sid].get("split", ""),
            "audio_path": os.path.abspath(wav),
            "transcript": trans.get(sid, ""),
        })
    return out, missing_meta, missing_audio


def labeler_dir(labeler):
    return os.path.join(LABELS_DIR, "by_labeler", labeler)


def labels_file(labeler, metric):
    return os.path.join(labeler_dir(labeler), f"{metric}.json")


def load_labels(labeler, metric):
    path = labels_file(labeler, metric)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def save_labels(labeler, metric, labels):
    os.makedirs(labeler_dir(labeler), exist_ok=True)
    tmp = labels_file(labeler, metric) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(labels, f, indent=2)
    os.replace(tmp, labels_file(labeler, metric))


HTML_TEMPLATE = string.Template("""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Label (labeler: $labeler)</title>
<style>
  body { font-family: system-ui; max-width: 700px; margin: 40px auto; padding: 0 20px; background: #1a1a2e; color: #eee; }
  h1 { color: #e94560; font-size: 1.25em; }
  .progress { color: #9aa; margin-bottom: 20px; }
  .done { color: #4f4; font-size: 1.3em; }
  audio { width: 100%; margin: 15px 0; }
  .info { background: #16213e; padding: 12px; border-radius: 8px; margin: 10px 0; font-size: 0.9em; }
  .field { margin: 18px 0; }
  .field label { display: block; font-weight: bold; margin-bottom: 4px; font-size: 1.1em; }
  .field .hint { color: #9aa; font-size: 0.85em; margin-bottom: 8px; }
  .slider-row { display: flex; align-items: center; gap: 12px; }
  input[type=range] { flex: 1; accent-color: #e94560; height: 6px; }
  .val { font-size: 1.5em; font-weight: bold; color: #e94560; min-width: 30px; text-align: center; }
  button { background: #e94560; color: #fff; border: none; padding: 12px 32px; font-size: 1.1em;
           border-radius: 6px; cursor: pointer; margin-top: 20px; }
  button:hover { background: #c73650; }
  .skip { background: #333; margin-left: 10px; }
  .skip:hover { background: #555; }
</style></head><body>
<h1>Voice clip labeling &mdash; labeler: <b>$labeler</b></h1>
<div class="progress">$progress</div>
$content
</body></html>""")

FORM_TEMPLATE = string.Template("""
<div class="info"><strong>ID:</strong> $sample_id &nbsp;|&nbsp; <strong>Split:</strong> $split</div>
<audio id="a" controls autoplay src="/audio/$audio_file"></audio>
<form method="POST" action="/label">
  <input type="hidden" name="sample_id" value="$sample_id">
  $fields
  <button type="submit">Save &amp; Next</button>
  <button type="submit" name="action" value="skip" class="skip">Skip</button>
</form>""")

FIELD_TEMPLATE = string.Template("""
  <div class="field">
    <label>$metric_cap</label>
    <div class="hint">$hint</div>
    <div class="slider-row">
      <span>1</span>
      <input type="range" name="$metric" min="1" max="10" value="5"
             oninput="document.getElementById('sv_$metric').textContent=this.value">
      <span>10</span>
      <span class="val" id="sv_$metric">5</span>
    </div>
  </div>""")

# Single-metric mode: big 1-10 buttons + keyboard shortcuts for fast labeling.
FAST_FORM_TEMPLATE = string.Template("""
<div class="info"><strong>ID:</strong> $sample_id &nbsp;|&nbsp; <strong>Split:</strong> $split</div>
<audio id="a" controls autoplay src="/audio/$audio_file"></audio>
<div class="field"><label>$metric_cap</label><div class="hint">$hint</div></div>
<div class="scale" id="scale"></div>
<div class="ends"><span>1</span><span>10</span></div>
<div class="row"><button type="button" onclick="document.getElementById('a').play()">Replay (R)</button>
<button type="button" class="skip" onclick="skip()">Skip (S)</button></div>
<script>
const SID="$sample_id", METRIC="$metric";
let busy=false;
const sc=document.getElementById('scale');
for(let i=1;i<=10;i++){const b=document.createElement('button');b.textContent=i;
  b.onclick=()=>rate(i);sc.appendChild(b);}
async function rate(v){ if(busy)return; busy=true;
  await fetch('/label',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},
    body:'sample_id='+encodeURIComponent(SID)+'&'+METRIC+'='+v}); location.reload(); }
async function skip(){ if(busy)return; busy=true;
  await fetch('/label',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},
    body:'sample_id='+encodeURIComponent(SID)+'&action=skip'}); location.reload(); }
document.addEventListener('keydown',e=>{
  if(e.key>='1'&&e.key<='9')rate(+e.key); else if(e.key==='0')rate(10);
  else if(e.key==='r'||e.key==='R'){const a=document.getElementById('a');if(a)a.play();}
  else if(e.key==='s'||e.key==='S')skip();});
</script>
<style>
 .scale{display:flex;gap:6px;flex-wrap:wrap}
 .scale button{flex:1;min-width:52px;padding:16px 0;font-size:1.3em;background:#0f3460;color:#fff;
   border:0;border-radius:6px;cursor:pointer;margin:0} .scale button:hover{background:#e94560}
 .ends{display:flex;justify-content:space-between;color:#9aa;font-size:.85em;margin:6px 2px 14px}
 .row button{margin-top:0}
</style>""")


class LabelHandler(SimpleHTTPRequestHandler):
    samples = []
    labeler = "labeler"
    metrics = METRICS             # which metrics this run collects
    labeled_ids = set()           # ids that have ALL this run's metrics (or were skipped)

    def _next_unlabeled(self):
        for s in self.samples:
            if s["id"] not in self.labeled_ids:
                return s
        return None

    def _send_html(self, html, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(html.encode())

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/audio/"):
            filename = os.path.basename(parsed.path[7:])
            audio_path = os.path.join(SAMPLES_DIR, "audio", filename)
            if os.path.exists(audio_path):
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.end_headers()
                with open(audio_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404)
            return

        total = len(self.samples)
        sample = self._next_unlabeled()
        if sample is None:
            content = '<div class="done">All samples labeled. Thank you!</div>'
        elif len(self.metrics) == 1:
            m = self.metrics[0]
            content = FAST_FORM_TEMPLATE.substitute(
                sample_id=sample["id"],
                split=sample["split"],
                audio_file=f"{sample['id']}.wav",
                metric=m, metric_cap=m.capitalize(), hint=METRIC_HINTS[m],
            )
        else:
            fields = "".join(
                FIELD_TEMPLATE.substitute(
                    metric=m, metric_cap=m.capitalize(), hint=METRIC_HINTS[m]
                )
                for m in self.metrics
            )
            content = FORM_TEMPLATE.substitute(
                sample_id=sample["id"],
                split=sample["split"],
                audio_file=f"{sample['id']}.wav",
                fields=fields,
            )
        self._send_html(HTML_TEMPLATE.substitute(
            labeler=self.labeler,
            progress=f"Labeled: {len(self.labeled_ids)} / {total}",
            content=content,
        ))

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        params = parse_qs(self.rfile.read(length).decode())
        sample_id = params.get("sample_id", [""])[0]
        action = params.get("action", ["save"])[0]

        if sample_id and action != "skip":
            sample = next((s for s in self.samples if s["id"] == sample_id), None)
            if sample:
                for metric in self.metrics:
                    raw = params.get(metric, ["5"])[0]
                    try:
                        score = int(raw)
                    except ValueError:
                        score = 5
                    score = max(1, min(10, score))
                    labels = load_labels(self.labeler, metric)
                    new_entry = {
                        "audio_path": sample["audio_path"],
                        "sample_id": sample_id,
                        "split": sample["split"],
                        metric: score,
                    }
                    for i, existing in enumerate(labels):
                        if existing.get("sample_id") == sample_id:
                            labels[i] = new_entry
                            break
                    else:
                        labels.append(new_entry)
                    save_labels(self.labeler, metric, labels)
                self.labeled_ids.add(sample_id)
        elif sample_id and action == "skip":
            self.labeled_ids.add(sample_id)

        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--labeler", required=True, help="labeler name (letters, digits, _ or -)")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--only-ids", default=None,
                        help="path to a JSON file of sample_ids to restrict labeling to "
                             "(e.g. data/training_dataset/labels/test76_ids.json)")
    parser.add_argument("--metric", choices=METRICS, default=None,
                        help="label a SINGLE metric per pass (faster: big 1-10 buttons + "
                             "number-key shortcuts). Default: all three on one screen.")
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", args.labeler):
        parser.error("--labeler may only contain letters, digits, _ and -")

    metrics = [args.metric] if args.metric else METRICS
    LabelHandler.metrics = metrics

    # Guard: never let a labeler name collide with the original labels directory.
    if args.only_ids:
        ids = load_only_ids(args.only_ids)
        samples, missing_meta, missing_audio = build_samples_for_ids(ids)
        if missing_meta:
            print(f"WARNING: {len(missing_meta)} id(s) not found in labels.json: "
                  f"{missing_meta[:5]}{' ...' if len(missing_meta) > 5 else ''}")
        if missing_audio:
            print(f"WARNING: {len(missing_audio)} id(s) have no wav in {AUDIO_DIR}: "
                  f"{missing_audio[:5]}{' ...' if len(missing_audio) > 5 else ''}")
        if not samples:
            parser.error(f"no usable clips matched the ids in {args.only_ids}")
        print(f"Restricted to {len(samples)}/{len(ids)} clips from {args.only_ids}")
    else:
        samples = load_samples()
    LabelHandler.samples = samples
    LabelHandler.labeler = args.labeler

    # A sample counts as done only if it has every metric for THIS run recorded.
    per_metric_ids = [{l["sample_id"] for l in load_labels(args.labeler, m)} for m in metrics]
    LabelHandler.labeled_ids = set.intersection(*per_metric_ids) if per_metric_ids else set()

    print(f"Labeler '{args.labeler}' [{', '.join(metrics)}]: "
          f"{len(LabelHandler.labeled_ids)}/{len(LabelHandler.samples)} clips done")
    print(f"Saving to {labeler_dir(args.labeler)}/<metric>.json (original labels untouched)")
    print(f"Open http://{args.host}:{args.port}  (Ctrl+C to stop; progress saved after every clip)")
    server = HTTPServer((args.host, args.port), LabelHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nStopped. {len(LabelHandler.labeled_ids)} clips labeled by '{args.labeler}'.")


if __name__ == "__main__":
    main()
