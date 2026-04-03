#!/usr/bin/env python3
"""
Web-based audio labeling tool for a SINGLE metric at a time.

Usage:
    python audio_quality_scoring/label_audio.py --metric naturalness [--port 8899]
    python audio_quality_scoring/label_audio.py --metric noisiness [--port 8900]
    python audio_quality_scoring/label_audio.py --metric loudness [--port 8901]

Labels saved to evaluation_data/training_dataset/labels/<metric>.json
"""

import json
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import argparse
import string

SAMPLES_DIR = "evaluation_data/training_dataset"
SAMPLES_JSON = os.path.join(SAMPLES_DIR, "samples.json")
LABELS_DIR = os.path.join(SAMPLES_DIR, "labels")

METRIC_HINTS = {
    "naturalness": "1 = very robotic/synthetic &nbsp;|&nbsp; 10 = perfectly human-like",
    "noisiness": "1 = very clean/clear &nbsp;|&nbsp; 10 = extremely noisy",
    "loudness": "1 = barely audible &nbsp;|&nbsp; 10 = very loud",
}


def labels_file(metric):
    return os.path.join(LABELS_DIR, f"{metric}.json")


def load_samples():
    with open(SAMPLES_JSON) as f:
        return json.load(f)["samples"]


def load_labels(metric):
    path = labels_file(metric)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def save_labels(metric, labels):
    os.makedirs(LABELS_DIR, exist_ok=True)
    with open(labels_file(metric), "w") as f:
        json.dump(labels, f, indent=2)


HTML_TEMPLATE = string.Template("""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Label: $metric</title>
<style>
  body { font-family: system-ui; max-width: 700px; margin: 40px auto; padding: 0 20px; background: #1a1a2e; color: #eee; }
  h1 { color: #e94560; }
  .progress { color: #888; margin-bottom: 20px; }
  .done { color: #0f0; font-size: 1.3em; }
  audio { width: 100%; margin: 15px 0; }
  .info { background: #16213e; padding: 12px; border-radius: 8px; margin: 10px 0; font-size: 0.9em; }
  .field { margin: 18px 0; }
  .field label { display: block; font-weight: bold; margin-bottom: 6px; font-size: 1.2em; }
  .field .hint { color: #888; font-size: 0.85em; margin-bottom: 8px; }
  .slider-row { display: flex; align-items: center; gap: 12px; }
  input[type=range] { flex: 1; accent-color: #e94560; height: 6px; }
  .val { font-size: 1.6em; font-weight: bold; color: #e94560; min-width: 30px; text-align: center; }
  button { background: #e94560; color: #fff; border: none; padding: 12px 32px; font-size: 1.1em;
           border-radius: 6px; cursor: pointer; margin-top: 20px; }
  button:hover { background: #c73650; }
  .skip { background: #333; margin-left: 10px; }
  .skip:hover { background: #555; }
</style></head><body>
<h1>Label: $metric_cap</h1>
<div class="progress">$progress</div>
$content
</body></html>""")

FORM_TEMPLATE = string.Template("""
<div class="info">
  <strong>ID:</strong> $sample_id &nbsp;|&nbsp; <strong>Split:</strong> $split
</div>
<audio controls autoplay src="/audio/$audio_file"></audio>
<form method="POST" action="/label">
  <input type="hidden" name="sample_id" value="$sample_id">
  <div class="field">
    <label>$metric_cap</label>
    <div class="hint">$hint</div>
    <div class="slider-row">
      <span>1</span>
      <input type="range" name="score" min="1" max="10" value="5"
             oninput="document.getElementById('sv').textContent=this.value">
      <span>10</span>
      <span class="val" id="sv">5</span>
    </div>
  </div>
  <button type="submit">Save &amp; Next</button>
  <button type="submit" name="action" value="skip" class="skip">Skip</button>
</form>""")


class LabelHandler(SimpleHTTPRequestHandler):
    samples = []
    labeled_ids = set()
    metric = "naturalness"

    def _next_unlabeled(self):
        for s in self.samples:
            if s["id"] not in self.labeled_ids:
                return s
        return None

    def _send_html(self, html, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode())

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/audio/"):
            filename = parsed.path[7:]
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

        labeled = len(self.labeled_ids)
        total = len(self.samples)
        sample = self._next_unlabeled()

        if sample is None:
            content = f'<div class="done">✅ All samples labeled for {self.metric}!</div>'
        else:
            content = FORM_TEMPLATE.substitute(
                sample_id=sample["id"],
                split=sample["split"],
                audio_file=f"{sample['id']}.wav",
                metric_cap=self.metric.capitalize(),
                hint=METRIC_HINTS[self.metric],
            )

        self._send_html(HTML_TEMPLATE.substitute(
            metric=self.metric,
            metric_cap=self.metric.capitalize(),
            progress=f"Labeled: {labeled} / {total}",
            content=content,
        ))

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode()
        params = parse_qs(body)

        sample_id = params.get("sample_id", [""])[0]
        action = params.get("action", ["save"])[0]

        if action != "skip" and sample_id:
            sample = next((s for s in self.samples if s["id"] == sample_id), None)
            if sample:
                labels = load_labels(self.metric)
                labels.append({
                    "audio_path": sample["audio_path"],
                    "sample_id": sample_id,
                    "split": sample["split"],
                    self.metric: int(params.get("score", [5])[0]),
                })
                save_labels(self.metric, labels)
                self.labeled_ids.add(sample_id)
        elif action == "skip" and sample_id:
            self.labeled_ids.add(sample_id)

        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description="Label audio for a single metric")
    parser.add_argument("--metric", required=True, choices=["naturalness", "noisiness", "loudness"])
    parser.add_argument("--port", type=int, default=8899)
    args = parser.parse_args()

    LabelHandler.samples = load_samples()
    LabelHandler.metric = args.metric

    existing = load_labels(args.metric)
    LabelHandler.labeled_ids = {l["sample_id"] for l in existing}
    print(f"[{args.metric}] Loaded {len(LabelHandler.samples)} samples, {len(existing)} already labeled")

    server = HTTPServer(("0.0.0.0", args.port), LabelHandler)
    print(f"Labeling UI ({args.metric}) at http://0.0.0.0:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nSaved {len(load_labels(args.metric))} labels to {labels_file(args.metric)}")


if __name__ == "__main__":
    main()
