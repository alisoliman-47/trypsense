from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import numpy as np
import cv2

app = FastAPI(title="Trypsin Estimator")

@app.get("/health")
def health():
    return {"status": "ok"}

# Pretty upload UI at "/"
@app.get("/", response_class=HTMLResponse)
def home():
    return """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>Trypsin Estimator</title>
    <style>
      :root{
        --bg:#0b1220;
        --card:#0f1b33;
        --card2:#0c162b;
        --text:#e6eefc;
        --muted:#9fb2d6;
        --border:rgba(255,255,255,.10);
        --accent:#6ee7ff;
        --accent2:#a78bfa;
        --danger:#fb7185;
        --ok:#34d399;
        --shadow: 0 10px 35px rgba(0,0,0,.45);
      }
      *{box-sizing:border-box}
      body{
        margin:0;
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
        color:var(--text);
        background:
          radial-gradient(1200px 700px at 20% 10%, rgba(110,231,255,.18), transparent 60%),
          radial-gradient(900px 600px at 80% 30%, rgba(167,139,250,.16), transparent 60%),
          radial-gradient(900px 600px at 60% 90%, rgba(52,211,153,.08), transparent 60%),
          linear-gradient(180deg, #070b14 0%, var(--bg) 50%, #050812 100%);
        min-height:100vh;
      }
      .wrap{max-width:1100px;margin:0 auto;padding:44px 20px 64px}
      .header{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;margin-bottom:18px}
      h1{margin:0;font-size:28px;letter-spacing:.2px}
      .sub{margin:6px 0 0;color:var(--muted);font-size:14px}
      .links{display:flex;gap:10px;flex-wrap:wrap}
      .chip{
        display:inline-flex;align-items:center;gap:8px;
        padding:8px 10px;border:1px solid var(--border);border-radius:999px;
        color:var(--text);text-decoration:none;font-size:13px;background:rgba(255,255,255,.03)
      }
      .chip:hover{border-color:rgba(255,255,255,.18);background:rgba(255,255,255,.05)}
      .grid{display:grid;grid-template-columns: 420px 1fr;gap:16px;margin-top:16px}
      @media (max-width: 920px){.grid{grid-template-columns:1fr}}
      .card{
        background: linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,.02));
        border:1px solid var(--border);
        border-radius:18px;
        box-shadow: var(--shadow);
        overflow:hidden;
      }
      .card .inner{padding:16px}
      .title{font-weight:650;margin:0 0 10px}
      .hint{color:var(--muted);font-size:13px;line-height:1.45;margin:0 0 14px}
      .drop{
        border:1.5px dashed rgba(255,255,255,.22);
        border-radius:16px;
        padding:14px;
        background: rgba(255,255,255,.02);
        transition: .15s ease;
      }
      .drop.drag{border-color: rgba(110,231,255,.55); background: rgba(110,231,255,.06)}
      .row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
      input[type=file]{display:none}
      .btn{
        appearance:none;border:1px solid var(--border);background:rgba(255,255,255,.04);
        color:var(--text);padding:10px 12px;border-radius:12px;cursor:pointer;
        font-weight:600;letter-spacing:.2px;font-size:14px;
      }
      .btn:hover{background:rgba(255,255,255,.07);border-color:rgba(255,255,255,.18)}
      .btn.primary{
        border-color: rgba(110,231,255,.35);
        background: linear-gradient(135deg, rgba(110,231,255,.22), rgba(167,139,250,.18));
      }
      .btn.primary:hover{border-color: rgba(110,231,255,.55)}
      .btn:disabled{opacity:.55;cursor:not-allowed}
      .meta{margin-top:10px;color:var(--muted);font-size:12px}
      .preview{
        margin-top:14px;
        border-radius:16px;
        overflow:hidden;
        border:1px solid rgba(255,255,255,.10);
        background: rgba(0,0,0,.25);
        height: 280px;
        display:flex;align-items:center;justify-content:center;
      }
      .preview img{max-width:100%;max-height:100%;display:none}
      .preview .placeholder{color:rgba(159,178,214,.85);font-size:13px;padding:18px;text-align:center}
      .result{
        display:flex;flex-direction:column;gap:10px;
      }
      .big{
        font-size:44px;font-weight:750;letter-spacing:.4px;margin:2px 0 0;
      }
      .label{color:var(--muted);font-size:13px;margin:0}
      .bar{
        height:12px;border-radius:999px;background: rgba(255,255,255,.08);
        border:1px solid rgba(255,255,255,.10);
        overflow:hidden;
      }
      .bar > div{
        height:100%;
        width:0%;
        background: linear-gradient(90deg, rgba(110,231,255,.95), rgba(167,139,250,.95));
        border-radius:999px;
        transition: width .35s ease;
      }
      .box{
        border:1px solid rgba(255,255,255,.10);
        background: rgba(0,0,0,.18);
        border-radius:16px;
        padding:14px;
      }
      .kv{display:grid;grid-template-columns: 160px 1fr;gap:10px;font-size:14px}
      .k{color:var(--muted)}
      .v{color:var(--text);word-break:break-word}
      .status{font-size:13px;margin-top:8px}
      .status.ok{color:var(--ok)}
      .status.err{color:var(--danger)}
      .footer{margin-top:18px;color:rgba(159,178,214,.9);font-size:12px;line-height:1.45}
      code{font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace}
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="header">
        <div>
          <h1>Trypsin Estimator</h1>
          <p class="sub">Upload a grid image to get <b>darkness %</b> based on dark spots on the grid.</p>
        </div>
        <div class="links">
          <a class="chip" href="/docs">API Docs</a>
          <a class="chip" href="/redoc">ReDoc</a>
          <a class="chip" href="/health">Health</a>
        </div>
      </div>

      <div class="grid">
        <div class="card">
          <div class="inner">
            <p class="title">Upload image</p>
            <p class="hint">Drop a photo here or choose a file. We only analyze the central grid region and return a percentage (clean grid should be ~0%).</p>

            <div id="drop" class="drop">
              <div class="row">
                <label class="btn" for="file">Choose file</label>
                <button id="analyze" class="btn primary" disabled>Analyze</button>
                <button id="clear" class="btn" disabled>Clear</button>
                <input id="file" type="file" accept="image/*" />
              </div>
              <div id="fileMeta" class="meta">No file selected.</div>

              <div class="preview">
                <div id="placeholder" class="placeholder">
                  Tip: use the same lighting/zoom for consistent results.<br/>
                  Your image stays on your machine; it’s only sent to this local server.
                </div>
                <img id="imgPreview" alt="preview" />
              </div>
            </div>

            <div id="status" class="status"></div>
            <div class="footer">
              Endpoint used: <code>POST /predict</code> with form field <code>file</code>.
            </div>
          </div>
        </div>

        <div class="card">
          <div class="inner result">
            <p class="title">Result</p>
            <div class="box">
              <p class="label">Darkness</p>
              <div class="big"><span id="darkness">—</span><span style="font-size:18px;font-weight:650;color:var(--muted)">%</span></div>
              <div class="bar"><div id="barFill"></div></div>
              <div style="height:10px"></div>
              <div class="kv">
                <div class="k">Fraction</div><div class="v" id="fraction">—</div>
                <div class="k">Trypsin (placeholder)</div><div class="v" id="trypsin">—</div>
                <div class="k">Notes</div><div class="v" id="notes">Upload an image to begin.</div>
              </div>
            </div>
            <div class="footer">
              If your “clear” images don’t read ~0%, we can tune the thresholding (or detect the circular grid more precisely instead of a center crop).
            </div>
          </div>
        </div>
      </div>
    </div>

    <script>
      const fileInput = document.getElementById('file');
      const drop = document.getElementById('drop');
      const analyzeBtn = document.getElementById('analyze');
      const clearBtn = document.getElementById('clear');
      const fileMeta = document.getElementById('fileMeta');
      const statusEl = document.getElementById('status');
      const imgPreview = document.getElementById('imgPreview');
      const placeholder = document.getElementById('placeholder');

      const darknessEl = document.getElementById('darkness');
      const fractionEl = document.getElementById('fraction');
      const trypsinEl = document.getElementById('trypsin');
      const notesEl = document.getElementById('notes');
      const barFill = document.getElementById('barFill');

      let currentFile = null;

      function setStatus(text, kind){
        statusEl.textContent = text || '';
        statusEl.className = 'status' + (kind ? (' ' + kind) : '');
      }

      function resetResult(){
        darknessEl.textContent = '—';
        fractionEl.textContent = '—';
        trypsinEl.textContent = '—';
        notesEl.textContent = 'Upload an image to begin.';
        barFill.style.width = '0%';
      }

      function setFile(file){
        currentFile = file || null;
        analyzeBtn.disabled = !currentFile;
        clearBtn.disabled = !currentFile;

        if(!currentFile){
          fileMeta.textContent = 'No file selected.';
          imgPreview.style.display = 'none';
          placeholder.style.display = 'block';
          imgPreview.src = '';
          setStatus('', '');
          resetResult();
          return;
        }

        const mb = (currentFile.size / (1024*1024)).toFixed(2);
        fileMeta.textContent = `${currentFile.name} • ${mb} MB`;

        const url = URL.createObjectURL(currentFile);
        imgPreview.onload = () => URL.revokeObjectURL(url);
        imgPreview.src = url;
        imgPreview.style.display = 'block';
        placeholder.style.display = 'none';
        setStatus('Ready to analyze.', 'ok');
      }

      fileInput.addEventListener('change', (e) => {
        const f = e.target.files && e.target.files[0];
        setFile(f || null);
      });

      clearBtn.addEventListener('click', () => {
        fileInput.value = '';
        setFile(null);
      });

      // Drag & drop
      ['dragenter','dragover'].forEach(evt => drop.addEventListener(evt, (e) => {
        e.preventDefault(); e.stopPropagation();
        drop.classList.add('drag');
      }));
      ['dragleave','drop'].forEach(evt => drop.addEventListener(evt, (e) => {
        e.preventDefault(); e.stopPropagation();
        drop.classList.remove('drag');
      }));
      drop.addEventListener('drop', (e) => {
        const f = e.dataTransfer.files && e.dataTransfer.files[0];
        if(f) setFile(f);
      });

      analyzeBtn.addEventListener('click', async () => {
        if(!currentFile) return;
        setStatus('Analyzing…', '');
        analyzeBtn.disabled = true;

        try{
          const fd = new FormData();
          fd.append('file', currentFile);

          const res = await fetch('/predict', { method: 'POST', body: fd });
          const data = await res.json().catch(() => ({}));

          if(!res.ok || data.error){
            const msg = data.error || `Request failed (${res.status})`;
            setStatus(msg, 'err');
            analyzeBtn.disabled = false;
            return;
          }

          const pct = typeof data.darkness_percent === 'number' ? data.darkness_percent : null;
          const frac = typeof data.darkness_fraction === 'number' ? data.darkness_fraction : null;

          if(pct === null || frac === null){
            setStatus('Unexpected response format from server.', 'err');
            analyzeBtn.disabled = false;
            return;
          }

          const pctClamped = Math.max(0, Math.min(100, pct));
          darknessEl.textContent = pctClamped.toFixed(2);
          fractionEl.textContent = frac.toFixed(6);
          trypsinEl.textContent = (typeof data.trypsin_estimate === 'number') ? data.trypsin_estimate.toFixed(6) : String(data.trypsin_estimate);
          notesEl.textContent = data.notes || '';
          barFill.style.width = pctClamped + '%';
          setStatus('Done.', 'ok');
        }catch(err){
          setStatus(err?.message || 'Failed to analyze image.', 'err');
        }finally{
          analyzeBtn.disabled = !currentFile;
        }
      });
    </script>
  </body>
</html>
"""

# Example calibration (placeholder!)
# Replace with real calibration fitted from your standards.
# concentration = a * darkness + b
CAL_A = 120.0
CAL_B = 0.0

def estimate_darkness(img_bgr: np.ndarray) -> float:
    """
    Compute how much of the grid area is occupied by dark spots.

    Returns:
        darkness_fraction in [0, 1], where:
        - 0   = clean grid (no dark spots)
        - 1   = grid fully covered by dark spots
    """
    # 1) Convert to grayscale for intensity analysis
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2) Focus on the central grid region only (tight center crop)
    h, w = gray.shape
    y1, y2 = int(0.3 * h), int(0.7 * h)
    x1, x2 = int(0.3 * w), int(0.7 * w)
    crop = gray[y1:y2, x1:x2]

    if crop.size == 0:
        return 0.0

    # 3) Slight blur to reduce sensor noise
    crop_blur = cv2.GaussianBlur(crop, (5, 5), 0)

    # 4) Estimate the "clean grid" brightness from the bright pixels.
    #    On a clean grid, almost all pixels are bright and similar.
    #    Dark spots show up as a tail of darker pixels.
    bright_percentile = np.percentile(crop_blur, 90)  # background level
    std_estimate = np.std(crop_blur)

    # Threshold that defines a "dark spot" pixel
    # Anything significantly darker than the bright background is counted.
    threshold = bright_percentile - 1.5 * std_estimate
    threshold = max(threshold, 0.0)

    # 5) Count dark pixels within the grid crop
    dark_mask = crop_blur < threshold
    dark_pixels = int(np.count_nonzero(dark_mask))
    total_pixels = int(crop_blur.size)

    if total_pixels == 0:
        return 0.0

    darkness_fraction = dark_pixels / total_pixels  # 0..1
    return float(np.clip(darkness_fraction, 0.0, 1.0))

def darkness_to_trypsin(darkness: float) -> float:
    # Placeholder linear calibration
    return CAL_A * darkness + CAL_B

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Could not decode image"}

    darkness = estimate_darkness(img)
    trypsin = darkness_to_trypsin(darkness)

    return {
        "darkness_fraction": darkness,
        "darkness_percent": round(darkness * 100.0, 2),
        "trypsin_estimate": trypsin,
        "units": "YOUR_UNITS_HERE",
        "notes": "Darkness is based only on the central grid. Calibration is placeholder until you fit to your standards."
    }