# Trypsin Estimator

A small FastAPI app that estimates **grid darkness %** (as a proxy for trypsin amount) from an uploaded image.  
It focuses on the **central grid region** and returns:

- `darkness_percent` – 0–100% (0% ≈ clean grid, higher = more dark spots)
- `darkness_fraction` – 0–1
- `trypsin_estimate` – placeholder value (you can calibrate later)

---

## 1. Prerequisites

- Python 3.9+ installed
- macOS with `zsh` (default for this project)

---

## 2. Create and activate a virtual environment

From the project folder:

```bash
cd "/Users/asoliman/Desktop/Nano 4b/trypsense"

python3 -m venv .venv
source .venv/bin/activate
```

Your terminal prompt should now start with `(.venv)`.

---

## 3. Install dependencies

Inside the virtual environment:

```bash
pip install fastapi uvicorn opencv-python numpy
```

(Optionally add `pip install python-multipart` if FastAPI warns about it for file uploads.)

---

## 4. Run the app

From the project root, with the virtual environment activated:

```bash
uvicorn app:app --reload
```

- The server will start at: `http://127.0.0.1:8000`

---

## 5. Use the web UI

Open in your browser:

- `http://127.0.0.1:8000/` – **Pretty upload UI**
  - Drag & drop or choose an image file
  - Click **Analyze**
  - See:
    - `darkness_percent`
    - `darkness_fraction`
    - `trypsin_estimate`
    - Notes

---

## 6. Use the API directly

### Health check

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status": "ok"}
```

### Predict from terminal

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@/full/path/to/your_image.jpg"
```

### Interactive docs

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

---

## 7. Stopping the app

In the terminal running `uvicorn`, press:

```text
Ctrl + C
```

To leave the virtual environment:

```bash
deactivate
```

