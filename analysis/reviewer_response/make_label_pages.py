"""
Generate a standalone, self-contained HTML labeling page per annotator for the
blinded provider MOS study. No server required: open the .html file directly in
a browser. Audio is referenced relatively (clips/clip_XXXX.wav), so the HTML is
written INSIDE results/provider_mos/ next to the clips/ folder.

Features for fast labeling:
  * Autoplay + auto-advance on rating
  * Keyboard: 1-5 = rate, R = replay, Left/Right = prev/next, Space = play/pause
  * Live progress bar + resume (ratings cached in the browser localStorage)
  * Export CSV in the exact schema analyze_provider_mos.py reads
    (position, clip_id, audio_file, naturalness_1to5)
  * Blinded: page shows only opaque clip IDs; no provider info is embedded

Run:
  /home/ubuntu/VoiceAIStackEvaluation/.venv/bin/python make_label_pages.py
"""
from __future__ import annotations

import csv
import glob
import json
import os

MOS_DIR = os.path.join(os.path.dirname(__file__), "results", "provider_mos")

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Naturalness Labeling &mdash; {ann}</title>
<style>
  :root {{ --bg:#0f1221; --card:#1b1f36; --fg:#eef1ff; --muted:#9aa3c7;
          --accent:#5b8cff; --good:#38c172; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:system-ui,Segoe UI,Roboto,sans-serif;
         background:var(--bg); color:var(--fg); }}
  header {{ padding:14px 20px; background:var(--card); position:sticky; top:0;
           box-shadow:0 2px 10px rgba(0,0,0,.4); }}
  h1 {{ font-size:16px; margin:0 0 8px; }}
  .bar {{ height:8px; background:#2a2f4d; border-radius:6px; overflow:hidden; }}
  .bar > div {{ height:100%; width:0; background:var(--good); transition:width .2s; }}
  .wrap {{ max-width:720px; margin:40px auto; padding:0 20px; text-align:center; }}
  .clip {{ font-size:14px; color:var(--muted); letter-spacing:.5px; }}
  .counter {{ font-size:13px; color:var(--muted); margin-top:4px; }}
  button.play {{ margin:26px 0; padding:16px 30px; font-size:18px; border:0;
                border-radius:12px; background:var(--accent); color:#fff;
                cursor:pointer; }}
  .scale {{ display:flex; gap:10px; justify-content:center; margin-top:14px; }}
  .scale button {{ flex:1; max-width:120px; padding:16px 0; border:1px solid #333a5e;
    border-radius:12px; background:var(--card); color:var(--fg); cursor:pointer;
    font-size:16px; }}
  .scale button b {{ display:block; font-size:22px; margin-bottom:4px; }}
  .scale button:hover {{ border-color:var(--accent); }}
  .scale button.sel {{ background:var(--accent); border-color:var(--accent); }}
  .hint {{ margin-top:22px; font-size:12px; color:var(--muted); line-height:1.6; }}
  .nav {{ margin-top:18px; }}
  .nav button {{ background:transparent; color:var(--muted); border:1px solid #333a5e;
    border-radius:8px; padding:8px 14px; cursor:pointer; margin:0 4px; }}
  .export {{ margin-top:26px; }}
  .export button {{ background:var(--good); color:#fff; border:0; border-radius:10px;
    padding:12px 22px; font-size:15px; cursor:pointer; }}
  .done {{ color:var(--good); font-weight:600; }}
</style>
</head>
<body>
<header>
  <h1>Naturalness labeling &mdash; annotator: {ann}</h1>
  <div class="bar"><div id="prog"></div></div>
</header>
<div class="wrap">
  <div class="clip" id="clipid">&mdash;</div>
  <div class="counter"><span id="pos">0</span> / <span id="total">0</span>
      &nbsp;&middot;&nbsp; rated: <span id="rated">0</span></div>
  <audio id="au"></audio>
  <div><button class="play" id="playbtn">&#9654; Play (Space)</button></div>
  <div class="scale" id="scale"></div>
  <div class="nav">
    <button id="prev">&larr; Prev</button>
    <button id="next">Next &rarr;</button>
    <button id="replay">Replay (R)</button>
  </div>
  <div class="export">
    <button id="exportbtn">Export CSV</button>
    <span id="savemsg"></span>
  </div>
  <div class="hint">
    5-point naturalness (ACR): <b>1</b> Bad &middot; <b>2</b> Poor &middot;
    <b>3</b> Fair &middot; <b>4</b> Good &middot; <b>5</b> Excellent.<br>
    Keys: <b>1&ndash;5</b> rate &amp; advance &middot; <b>R</b> replay &middot;
    <b>&larr;/&rarr;</b> move &middot; <b>Space</b> play/pause.<br>
    Ratings auto-save in this browser. Click <b>Export CSV</b> when finished and
    send the file back.
  </div>
</div>
<script>
const ANN = {ann_json};
const CLIPS = {clips_json};   // [{{position, clip_id, audio_file}}]
const KEY = "provmos_" + ANN;
let ratings = JSON.parse(localStorage.getItem(KEY) || "{{}}");
let i = 0;

const au = document.getElementById("au");
const scale = document.getElementById("scale");
const LABELS = {{1:"Bad",2:"Poor",3:"Fair",4:"Good",5:"Excellent"}};
for (let v=1; v<=5; v++) {{
  const b=document.createElement("button");
  b.innerHTML = "<b>"+v+"</b>"+LABELS[v];
  b.onclick=()=>rate(v);
  b.dataset.v=v; scale.appendChild(b);
}}
document.getElementById("total").textContent = CLIPS.length;

function render() {{
  const c = CLIPS[i];
  document.getElementById("clipid").textContent = c.clip_id;
  document.getElementById("pos").textContent = (i+1);
  au.src = c.audio_file;
  au.play().catch(()=>{{}});
  [...scale.children].forEach(b=>b.classList.toggle("sel",
      ratings[c.clip_id]==b.dataset.v));
  const n = Object.keys(ratings).length;
  document.getElementById("rated").textContent = n;
  document.getElementById("prog").style.width = (100*n/CLIPS.length)+"%";
}}
function rate(v) {{
  ratings[CLIPS[i].clip_id] = v;
  localStorage.setItem(KEY, JSON.stringify(ratings));
  render();
  setTimeout(()=>{{ if (i < CLIPS.length-1) {{ i++; render(); }} }}, 180);
}}
function go(d) {{ i = Math.max(0, Math.min(CLIPS.length-1, i+d)); render(); }}

document.getElementById("playbtn").onclick=()=>au.paused?au.play():au.pause();
document.getElementById("replay").onclick=()=>{{au.currentTime=0;au.play();}};
document.getElementById("prev").onclick=()=>go(-1);
document.getElementById("next").onclick=()=>go(1);
document.addEventListener("keydown",e=>{{
  if (e.key>="1" && e.key<="5") rate(parseInt(e.key));
  else if (e.key.toLowerCase()==="r") {{au.currentTime=0;au.play();}}
  else if (e.key==="ArrowLeft") go(-1);
  else if (e.key==="ArrowRight") go(1);
  else if (e.key===" ") {{e.preventDefault(); au.paused?au.play():au.pause();}}
}});
document.getElementById("exportbtn").onclick=()=>{{
  let rows=[["position","clip_id","audio_file","naturalness_1to5"]];
  CLIPS.forEach(c=>rows.push([c.position,c.clip_id,c.audio_file,
      ratings[c.clip_id]||""]));
  const csv=rows.map(r=>r.join(",")).join("\\n");
  const blob=new Blob([csv],{{type:"text/csv"}});
  const a=document.createElement("a");
  a.href=URL.createObjectURL(blob);
  a.download="playlist_"+ANN+".csv";
  a.click();
  const n=Object.keys(ratings).length;
  document.getElementById("savemsg").innerHTML =
    " <span class='done'>exported "+n+"/"+CLIPS.length+"</span>";
}};
render();
</script>
</body>
</html>
"""


def main():
    playlists = sorted(glob.glob(os.path.join(MOS_DIR, "playlist_*.csv")))
    if not playlists:
        raise SystemExit("No playlists found. Run build_provider_mos_sample.py first.")
    made = []
    for pl in playlists:
        ann = os.path.basename(pl)[len("playlist_"):-len(".csv")]
        clips = []
        with open(pl) as f:
            for row in csv.DictReader(f):
                clips.append({"position": int(row["position"]),
                              "clip_id": row["clip_id"],
                              "audio_file": row["audio_file"]})
        html = PAGE.format(ann=ann, ann_json=json.dumps(ann),
                           clips_json=json.dumps(clips))
        out = os.path.join(MOS_DIR, f"label_{ann}.html")
        with open(out, "w") as f:
            f.write(html)
        made.append((out, len(clips)))
    print("Generated labeling pages (open directly in a browser):")
    for out, n in made:
        print(f"  {out}  ({n} clips)")
    print("\nEach annotator: open their page, rate with keys 1-5, then click "
          "'Export CSV' and save the downloaded playlist_<name>.csv back into")
    print(f"  {MOS_DIR}/  (overwriting the empty template), then run "
          "analyze_provider_mos.py.")


if __name__ == "__main__":
    main()
