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
          <p class="sub">Upload a grid image pair to get <b>darkness % vs your own baseline</b> based on dark spots on the inner grid.</p>
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
                <button id="analyzeBaseline" class="btn" disabled>Analyze as baseline</button>
                <button id="analyzeSample" class="btn primary" disabled>Analyze sample vs baseline</button>
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
              <p class="label">Sample vs baseline</p>
              <div class="big"><span id="darkness">—</span><span style="font-size:18px;font-weight:650;color:var(--muted)">%</span></div>
              <div class="bar"><div id="barFill"></div></div>
              <div style="height:10px"></div>
              <div class="kv">
                <div class="k">Baseline darkness</div><div class="v" id="baseline">—</div>
                <div class="k">Sample darkness (raw)</div><div class="v" id="fraction">—</div>
                <div class="k">Sample vs baseline</div><div class="v" id="relative">—</div>
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
      const analyzeBaselineBtn = document.getElementById('analyzeBaseline');
      const analyzeSampleBtn = document.getElementById('analyzeSample');
      const clearBtn = document.getElementById('clear');
      const fileMeta = document.getElementById('fileMeta');
      const statusEl = document.getElementById('status');
      const imgPreview = document.getElementById('imgPreview');
      const placeholder = document.getElementById('placeholder');

      const darknessEl = document.getElementById('darkness');
      const fractionEl = document.getElementById('fraction');
      const baselineEl = document.getElementById('baseline');
      const relativeEl = document.getElementById('relative');
      const trypsinEl = document.getElementById('trypsin');
      const notesEl = document.getElementById('notes');
      const barFill = document.getElementById('barFill');

      let currentFile = null;      // currently selected file (used as baseline *or* sample)
      let baselineFile = null;     // stored baseline image file

      function setStatus(text, kind){
        statusEl.textContent = text || '';
        statusEl.className = 'status' + (kind ? (' ' + kind) : '');
      }

      function resetResult(){
        darknessEl.textContent = '—';
        fractionEl.textContent = '—';
        baselineEl.textContent = '—';
        relativeEl.textContent = '—';
        trypsinEl.textContent = '—';
        notesEl.textContent = 'Step 1: upload a clear grid and set it as baseline. Step 2: upload a sample image with trypsin and analyze vs baseline.';
        barFill.style.width = '0%';
      }

      function setFile(file){
        currentFile = file || null;
        analyzeBaselineBtn.disabled = !currentFile;
        // sample analysis additionally requires a saved baseline
        analyzeSampleBtn.disabled = !currentFile || !baselineFile;
        clearBtn.disabled = !currentFile && !baselineFile;

        if(!currentFile){
          fileMeta.textContent = baselineFile ? 'No sample selected. Baseline is set.' : 'No file selected.';
          imgPreview.style.display = 'none';
          placeholder.style.display = 'block';
          imgPreview.src = '';
          setStatus('', '');
          resetResult();
          if(baselineFile){
            baselineEl.textContent = '0.00';
            relativeEl.textContent = '—';
            notesEl.textContent = 'Baseline is set. Upload a sample image and analyze vs baseline.';
          }
          return;
        }

        const mb = (currentFile.size / (1024*1024)).toFixed(2);
        fileMeta.textContent = `${currentFile.name} • ${mb} MB`;

        const url = URL.createObjectURL(currentFile);
        imgPreview.onload = () => URL.revokeObjectURL(url);
        imgPreview.src = url;
        imgPreview.style.display = 'block';
        placeholder.style.display = 'none';
        setStatus('Ready to analyze as baseline or sample.', 'ok');
      }

      fileInput.addEventListener('change', (e) => {
        const f = e.target.files && e.target.files[0];
        setFile(f || null);
      });

      clearBtn.addEventListener('click', () => {
        fileInput.value = '';
        currentFile = null;
        baselineFile = null;
        setFile(null);
        resetResult();
        setStatus('Cleared baseline and sample.', '');
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

      // Step 1: user chooses a clear grid image and marks it as baseline.
      analyzeBaselineBtn.addEventListener('click', () => {
        if(!currentFile) return;
        baselineFile = currentFile;
        baselineEl.textContent = '0.00';
        relativeEl.textContent = '—';
        barFill.style.width = '0%';
        notesEl.textContent = 'Baseline saved (treated as 0% darkness). Now choose a sample image (with trypsin) and analyze vs baseline.';
        setStatus('Baseline saved locally. It will be compared pixel‑by‑pixel when you analyze a sample.', 'ok');
        analyzeSampleBtn.disabled = !currentFile; // now enabled if a sample is present
      });

      // Step 2: user uploads a sample and we send baseline + sample together.
      analyzeSampleBtn.addEventListener('click', async () => {
        if(!currentFile || !baselineFile){
          setStatus('You need to set a baseline first, then choose a sample.', 'err');
          return;
        }
        setStatus('Analyzing sample vs baseline…', '');
        analyzeSampleBtn.disabled = true;

        try{
          const fd = new FormData();
          fd.append('baseline', baselineFile);
          fd.append('sample', currentFile);

          const res = await fetch('/predict_pair', { method: 'POST', body: fd });
          const data = await res.json().catch(() => ({}));

          if(!res.ok || data.error){
            const msg = data.error || `Request failed (${res.status})`;
            setStatus(msg, 'err');
            analyzeSampleBtn.disabled = false;
            return;
          }

          const basePct = data.baseline_darkness_percent;
          const baseFrac = data.baseline_darkness_fraction;
          const samplePct = data.sample_darkness_percent;
          const sampleFrac = data.sample_darkness_fraction;
          const relPct = data.relative_darkness_percent;
          const relFrac = data.relative_darkness_fraction;

          if(
            typeof samplePct !== 'number' ||
            typeof sampleFrac !== 'number' ||
            typeof relPct !== 'number' ||
            typeof relFrac !== 'number'
          ){
            setStatus('Unexpected response format from server.', 'err');
            analyzeSampleBtn.disabled = false;
            return;
          }

          const clampedSamplePct = Math.max(0, Math.min(100, samplePct));
          const clampedRelPct = Math.max(0, Math.min(100, relPct));

          // Top metric and bar show the relative (sample vs baseline) darkness.
          darknessEl.textContent = clampedRelPct.toFixed(2);
          fractionEl.textContent = sampleFrac.toFixed(6);
          baselineEl.textContent = (typeof basePct === 'number' ? basePct : 0).toFixed(2);
          relativeEl.textContent = clampedRelPct.toFixed(2);
          trypsinEl.textContent = (typeof data.trypsin_estimate === 'number') ? data.trypsin_estimate.toFixed(6) : String(data.trypsin_estimate);
          notesEl.textContent = data.notes || 'Relative darkness is based only on pixels that became darker than the baseline inside the inner grid.';
          barFill.style.width = clampedRelPct + '%';
          setStatus('Sample analyzed vs baseline.', 'ok');
        }catch(err){
          setStatus(err?.message || 'Failed to analyze sample vs baseline.', 'err');
        }finally{
          analyzeSampleBtn.disabled = false;
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


def _extract_grid_region(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the inner circular grid region as (crop_gray, mask).

    - crop_gray: grayscale crop roughly around the grid
    - mask: boolean array (same shape as crop_gray) where True marks
      pixels inside the detected circular grid.

    If circle detection fails, falls back to a simple center crop with
    a circular mask, still focusing on the inner region.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Start from a center crop to reduce noise from the outer area.
    y1, y2 = int(0.25 * h), int(0.75 * h)
    x1, x2 = int(0.25 * w), int(0.75 * w)
    crop = gray[y1:y2, x1:x2]
    ch, cw = crop.shape

    # Default: center circle mask in the crop.
    cy, cx = ch // 2, cw // 2
    r = min(ch, cw) * 0.35
    yy, xx = np.ogrid[:ch, :cw]
    base_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2

    # Try to refine using Hough circle detection on the crop.
    try:
        blur = cv2.medianBlur(crop, 5)
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min(ch, cw) / 2,
            param1=80,
            param2=20,
            minRadius=int(min(ch, cw) * 0.25),
            maxRadius=int(min(ch, cw) * 0.45),
        )
        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            c_x, c_y, c_r = circles[0]
            # Slightly shrink the radius to avoid the bright outer ring.
            c_r = int(c_r * 0.9)
            yy2, xx2 = np.ogrid[:ch, :cw]
            mask = (yy2 - c_y) ** 2 + (xx2 - c_x) ** 2 <= c_r ** 2
            return crop, mask
    except Exception:
        # Fall back gracefully if anything goes wrong.
        pass

    return crop, base_mask


def estimate_darkness(img_bgr: np.ndarray) -> float:
    """
    Single-image darkness estimate for the inner grid.

    This is mostly for debugging. The preferred, grid-line‑invariant
    metric is the relative darkness returned by compare_baseline_and_sample.
    """
    crop, mask = _extract_grid_region(img_bgr)
    if crop.size == 0:
        return 0.0

    crop_blur = cv2.GaussianBlur(crop, (5, 5), 0)
    roi = crop_blur[mask]
    if roi.size == 0:
        return 0.0

    # Use the bright end of the histogram as a reference background.
    bright_percentile = np.percentile(roi, 95)
    std_estimate = np.std(roi)
    threshold = bright_percentile - 2.0 * std_estimate
    threshold = max(threshold, 0.0)

    dark_mask = (crop_blur < threshold) & mask
    dark_pixels = int(np.count_nonzero(dark_mask))
    total_pixels = int(np.count_nonzero(mask))
    if total_pixels == 0:
        return 0.0

    darkness_fraction = dark_pixels / total_pixels
    return float(np.clip(darkness_fraction, 0.0, 1.0))


def compare_baseline_and_sample(
    baseline_bgr: np.ndarray, sample_bgr: np.ndarray
) -> float:
    """
    Compute relative darkness between a baseline (clear grid) and a sample.

    Uses only the inner circular grid region and focuses on pixels that
    become darker in the sample than in the baseline, which helps ignore
    the regular grid lines that appear in both images.
    """
    base_crop, base_mask = _extract_grid_region(baseline_bgr)
    sample_crop, sample_mask = _extract_grid_region(sample_bgr)

    if base_crop.size == 0 or sample_crop.size == 0:
        return 0.0

    # Resize to common size in case framing changed slightly between shots.
    h = min(base_crop.shape[0], sample_crop.shape[0])
    w = min(base_crop.shape[1], sample_crop.shape[1])
    if h <= 0 or w <= 0:
        return 0.0

    base = cv2.resize(base_crop, (w, h), interpolation=cv2.INTER_AREA)
    sample = cv2.resize(sample_crop, (w, h), interpolation=cv2.INTER_AREA)

    bmask = cv2.resize(
        base_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    smask = cv2.resize(
        sample_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    region_mask = bmask & smask
    if not np.any(region_mask):
        return 0.0

    base_blur = cv2.GaussianBlur(base, (5, 5), 0)
    sample_blur = cv2.GaussianBlur(sample, (5, 5), 0)

    # Positive difference = pixels that became darker compared to baseline.
    delta = base_blur.astype(np.float32) - sample_blur.astype(np.float32)
    delta[delta < 0] = 0.0

    # Apply inner-grid mask.
    delta_masked = delta[region_mask]
    if delta_masked.size == 0:
        return 0.0

    positive = delta_masked[delta_masked > 0]
    if positive.size == 0:
        return 0.0

    # Threshold on the distribution of positive deltas to ignore tiny fluctuations.
    mean_delta = float(np.mean(positive))
    std_delta = float(np.std(positive))
    threshold = max(mean_delta + 0.5 * std_delta, float(np.percentile(positive, 70)))

    strong = delta_masked >= threshold
    if not np.any(strong):
        return 0.0

    # Fraction of inner‑grid pixels that significantly darkened
    dark_pixels = int(np.count_nonzero(strong))
    total_pixels = int(delta_masked.size)
    if total_pixels == 0:
        return 0.0

    rel_fraction = dark_pixels / total_pixels  # 0..1
    return float(np.clip(rel_fraction, 0.0, 1.0))


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
        "notes": "Darkness is based only on the central grid. Calibration is placeholder until you fit to your standards.",
    }


@app.post("/predict_pair")
async def predict_pair(
    baseline: UploadFile = File(...),
    sample: UploadFile = File(...),
):
    """
    Compare a baseline (clear grid) image and a sample (with trypsin).

    - Baseline is treated as 0% darkness by definition.
    - Relative darkness only counts pixels inside the inner grid that
      became darker than baseline, largely ignoring the regular grid lines.
    """
    base_data = await baseline.read()
    sample_data = await sample.read()

    base_arr = np.frombuffer(base_data, np.uint8)
    sample_arr = np.frombuffer(sample_data, np.uint8)
    base_img = cv2.imdecode(base_arr, cv2.IMREAD_COLOR)
    sample_img = cv2.imdecode(sample_arr, cv2.IMREAD_COLOR)

    if base_img is None or sample_img is None:
        return {"error": "Could not decode one or both images"}

    # Relative darkness that ignores the regular grid pattern.
    rel_fraction = compare_baseline_and_sample(base_img, sample_img)

    # Optional: raw sample darkness using the single-image estimator.
    sample_fraction = estimate_darkness(sample_img)
    trypsin = darkness_to_trypsin(sample_fraction)

    return {
        "baseline_darkness_fraction": 0.0,
        "baseline_darkness_percent": 0.0,
        "sample_darkness_fraction": float(sample_fraction),
        "sample_darkness_percent": round(sample_fraction * 100.0, 2),
        "relative_darkness_fraction": float(rel_fraction),
        "relative_darkness_percent": round(rel_fraction * 100.0, 2),
        "trypsin_estimate": trypsin,
        "units": "YOUR_UNITS_HERE",
        "notes": "Relative darkness is computed only from pixels that become darker than the baseline within the inner grid.",
    }