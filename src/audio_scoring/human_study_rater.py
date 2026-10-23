#!/usr/bin/env python3
# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT
"""
Blinded naturalness-rating UI for the human study (one process per rater).

Usage:
    python src/audio_scoring/human_study_rater.py --rater alice [--port 8899]

Raters see only an anonymous clip number, an audio player and the 1-10 buttons.
Provider, prompt text and all metric scores stay on the server. Clip order is
shuffled per rater. Ratings are appended to data/human_study/ratings/<rater>.jsonl
after every click, so progress is kept if the browser closes; re-run the same
command to resume.

Security: the server has no authentication. It binds to 127.0.0.1 by default.
To let a rater on another machine use it, prefer an SSH tunnel
(ssh -L 8899:localhost:8899 host) over --host 0.0.0.0.
"""

import argparse
import hashlib
import json
import os
import random
import re
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

STUDY_DIR = os.environ.get("HUMAN_STUDY_DIR", "data/human_study")

PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Voice rating</title>
<style>
 body{font-family:system-ui;max-width:720px;margin:30px auto;padding:0 20px;background:#1a1a2e;color:#eee}
 h1{color:#e94560;font-size:1.3em} .prog{color:#9aa;margin:8px 0 16px}
 .guide{background:#16213e;padding:12px 16px;border-radius:8px;font-size:.92em;line-height:1.5}
 audio{width:100%;margin:18px 0}
 .scale{display:flex;gap:6px;flex-wrap:wrap} .scale button{flex:1;min-width:52px;padding:16px 0;font-size:1.3em;
   background:#0f3460;color:#fff;border:0;border-radius:6px;cursor:pointer} .scale button:hover{background:#e94560}
 .ends{display:flex;justify-content:space-between;color:#9aa;font-size:.85em;margin:6px 2px 18px}
 .row button{background:#333;color:#fff;border:0;padding:8px 16px;border-radius:6px;cursor:pointer;margin-right:8px}
 .done{color:#4f4;font-size:1.3em;margin-top:40px}
</style></head><body>
<h1>Rate the naturalness of each voice clip</h1>
<div class="guide">Judge <b>how human-like the voice sounds</b>: rhythm, intonation, emphasis and pacing.
Ignore <i>what</i> is said, background hiss and volume (volume is already equalised).
Use headphones in a quiet place. Listen to the whole clip, then click a number.
Keyboard: 1-9 and 0 (= 10), <b>R</b> replays, <b>U</b> undoes the last rating.</div>
<div class="prog" id="prog"></div>
<div id="main"></div>
<script>
let cur=null;
async function load(){
  const r=await (await fetch('/api/next')).json();
  document.getElementById('prog').textContent='Rated '+r.done+' / '+r.total;
  const m=document.getElementById('main');
  if(r.clip_id===null){m.innerHTML='<div class="done">All clips rated. Thank you!</div>';cur=null;return;}
  cur=r.clip_id;
  let b='';for(let i=1;i<=10;i++)b+='<button onclick="rate('+i+')">'+i+'</button>';
  m.innerHTML='<div>Clip '+(r.done+1)+'</div><audio id="a" controls autoplay src="/audio/'+cur+'.wav"></audio>'+
   '<div class="scale">'+b+'</div><div class="ends"><span>1 = very robotic</span><span>10 = indistinguishable from a human</span></div>'+
   '<div class="row"><button onclick="document.getElementById(\\'a\\').play()">Replay (R)</button>'+
   '<button onclick="undo()">Undo last (U)</button></div>';
}
async function rate(s){ if(cur===null)return;
  await fetch('/api/rate',{method:'POST',body:JSON.stringify({clip_id:cur,score:s})}); load(); }
async function undo(){ await fetch('/api/undo',{method:'POST'}); load(); }
document.addEventListener('keydown',e=>{
  if(e.key>='1'&&e.key<='9')rate(+e.key); else if(e.key==='0')rate(10);
  else if(e.key==='r'||e.key==='R'){const a=document.getElementById('a');if(a)a.play();}
  else if(e.key==='u'||e.key==='U')undo();});
load();
</script></body></html>"""


def build_order(clips, rater, min_gap=10):
    """Per-rater shuffle; keep each repeat at least `min_gap` positions from its original."""
    seed = int(hashlib.sha256(rater.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    ids = [c["clip_id"] for c in clips]
    repeat_of = {c["clip_id"]: c["repeat_of"] for c in clips if c["repeat_of"]}
    for _ in range(500):
        rng.shuffle(ids)
        pos = {cid: i for i, cid in enumerate(ids)}
        if all(abs(pos[r] - pos[o]) >= min_gap for r, o in repeat_of.items()):
            break
    return ids


class State:
    def __init__(self, rater):
        with open(os.path.join(STUDY_DIR, "manifest.json")) as f:
            self.clips = json.load(f)["clips"]
        self.valid = {c["clip_id"] for c in self.clips}
        self.order = build_order(self.clips, rater)
        os.makedirs(os.path.join(STUDY_DIR, "ratings"), exist_ok=True)
        self.path = os.path.join(STUDY_DIR, "ratings", f"{rater}.jsonl")
        self.ratings = {}      # clip_id -> score (latest wins; null = undone)
        self.history = []      # clip_ids in rating order, for undo
        if os.path.exists(self.path):
            with open(self.path) as f:
                for line in f:
                    e = json.loads(line)
                    if e["score"] is None:
                        self.ratings.pop(e["clip_id"], None)
                        if e["clip_id"] in self.history:
                            self.history.remove(e["clip_id"])
                    else:
                        self.ratings[e["clip_id"]] = e["score"]
                        self.history.append(e["clip_id"])

    def _append(self, clip_id, score):
        with open(self.path, "a") as f:
            f.write(json.dumps({"clip_id": clip_id, "score": score, "t": time.time()}) + "\n")

    def next_clip(self):
        for cid in self.order:
            if cid not in self.ratings:
                return cid
        return None

    def rate(self, clip_id, score):
        if clip_id in self.valid and isinstance(score, int) and 1 <= score <= 10:
            self.ratings[clip_id] = score
            self.history.append(clip_id)
            self._append(clip_id, score)

    def undo(self):
        if self.history:
            cid = self.history.pop()
            self.ratings.pop(cid, None)
            self._append(cid, None)


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, body, ctype="application/json", code=200):
            if isinstance(body, (dict, list)):
                body = json.dumps(body)
            if isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                return self._send(PAGE, "text/html; charset=utf-8")
            if self.path == "/api/next":
                return self._send({"clip_id": state.next_clip(), "done": len(state.ratings),
                                   "total": len(state.order)})
            m = re.fullmatch(r"/audio/(c[0-9a-f]{8})\.wav", self.path)
            if m and m.group(1) in state.valid:
                with open(os.path.join(STUDY_DIR, "audio", f"{m.group(1)}.wav"), "rb") as f:
                    return self._send(f.read(), "audio/wav")
            self._send("not found", "text/plain", 404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._send("bad request", "text/plain", 400)
            if self.path == "/api/rate":
                state.rate(body.get("clip_id"), body.get("score"))
            elif self.path == "/api/undo":
                state.undo()
            else:
                return self._send("not found", "text/plain", 404)
            self._send({"ok": True})

        def log_message(self, *a):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rater", required=True, help="rater name (letters, digits, _ or -)")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", args.rater):
        ap.error("--rater may only contain letters, digits, _ and -")
    state = State(args.rater)
    print(f"Rater '{args.rater}': {len(state.ratings)}/{len(state.order)} already rated")
    print(f"Open http://{args.host}:{args.port}  (Ctrl+C to stop; progress is saved after every click)")
    try:
        HTTPServer((args.host, args.port), make_handler(state)).serve_forever()
    except KeyboardInterrupt:
        print(f"\nSaved {len(state.ratings)} ratings to {state.path}")


if __name__ == "__main__":
    main()
