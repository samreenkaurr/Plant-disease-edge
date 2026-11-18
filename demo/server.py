"""Phone demo: your Mac runs the TFLite model, your iPhone takes the photo.

Run from the project root (with .venv active):
    pip install flask
    python demo/server.py
Then open the printed address in iPhone Safari (phone and Mac on the same Wi-Fi).
"""
import io, json, math, socket, threading, time
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "app" / "assets"
SIZE = 224

labels = [l.strip() for l in (ASSETS / "labels.txt").read_text().splitlines() if l.strip()]
cal = json.loads((ASSETS / "calibration.json").read_text())
TEMP, THRESH = float(cal["temperature"]), float(cal["threshold"])


def make_interpreter():
    path = str(ASSETS / "model.tflite")
    try:
        it = tf.lite.Interpreter(model_path=path)
        it.allocate_tensors()
    except RuntimeError:  # XNNPACK can fail to prepare some models; fall back to plain kernels
        it = tf.lite.Interpreter(
            model_path=path,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        it.allocate_tensors()
    return it


interp = make_interpreter()
IN_IDX = interp.get_input_details()[0]["index"]
OUT_IDX = interp.get_output_details()[0]["index"]
lock = threading.Lock()
app = Flask(__name__)

PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Plant Doctor demo</title>
<style>
 body{font-family:-apple-system,system-ui,sans-serif;margin:0;padding:16px;max-width:520px;margin:auto;color:#1b1b1b}
 h1{font-size:1.3rem} img{width:100%;border-radius:12px;margin:12px 0}
 label.btn{display:block;text-align:center;background:#2e7d32;color:#fff;padding:14px;border-radius:12px;font-weight:600}
 input{display:none} .row{margin:10px 0} .bar{height:10px;background:#e0e0e0;border-radius:6px;overflow:hidden}
 .bar>div{height:100%;background:#2e7d32} .warn{background:#fff3cd;padding:12px;border-radius:10px;margin:10px 0}
 .small{color:#666;font-size:.8rem;margin-top:16px}
</style></head><body>
<h1>Plant Doctor (demo)</h1>
<label class="btn">Take or choose a leaf photo<input id="f" type="file" accept="image/*"></label>
<img id="preview" style="display:none">
<div id="out"></div>
<p class="small">Demo only: the photo is analysed by the model running on the Mac, not on this phone.
Screening aid, not a substitute for an agronomist. Supports tomato and potato leaves only.</p>
<script>
const f=document.getElementById('f'), out=document.getElementById('out'), prev=document.getElementById('preview');
f.onchange=async()=>{
  const file=f.files[0]; if(!file) return;
  prev.src=URL.createObjectURL(file); prev.style.display='block'; out.textContent='Analysing...';
  const fd=new FormData(); fd.append('image',file);
  try{
    const r=await fetch('/predict',{method:'POST',body:fd}); const d=await r.json();
    if(d.error){out.textContent=d.error;return;}
    let h=d.uncertain?'<div class="warn">Not sure. Retake the photo: one leaf, good light, fill the frame.</div>':'';
    for(const p of d.top){h+='<div class="row">'+p.label.replace(/_/g,' ')+' <b>'+Math.round(p.confidence*100)+'%</b><div class="bar"><div style="width:'+(p.confidence*100)+'%"></div></div></div>';}
    h+='<p class="small">Inference: '+d.latency_ms.toFixed(1)+' ms (on the Mac)</p>';
    out.innerHTML=h;
  }catch(e){out.textContent='Error: '+e;}
};
</script></body></html>"""


@app.get("/")
def index():
    return PAGE


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify(error="No image received")
    try:
        img = Image.open(io.BytesIO(request.files["image"].read()))
        img = ImageOps.exif_transpose(img).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    except Exception as e:
        return jsonify(error=f"Could not read that image: {e}")
    x = np.asarray(img, dtype=np.float32)[None]          # 0-255, the model normalises internally
    with lock:
        t0 = time.perf_counter()
        interp.set_tensor(IN_IDX, x)
        interp.invoke()
        probs = interp.get_tensor(OUT_IDX)[0].astype(np.float64)
        ms = (time.perf_counter() - t0) * 1000
    logits = np.log(np.clip(probs, 1e-8, 1.0)) / TEMP    # temperature scaling, same as the app code
    e = np.exp(logits - logits.max())
    p = e / e.sum()
    order = np.argsort(-p)[:3]
    top = [{"label": labels[i], "confidence": float(p[i])} for i in order]
    return jsonify(top=top, uncertain=bool(p[order[0]] < THRESH), latency_ms=ms)


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


if __name__ == "__main__":
    print(f"\nOpen this on your iPhone (same Wi-Fi):  http://{lan_ip()}:8000\n")
    app.run(host="0.0.0.0", port=8000, threaded=True)
