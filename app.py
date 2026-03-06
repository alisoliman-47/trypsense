from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2

app = FastAPI(title="Trypsin Estimator")

# Example calibration (placeholder!)
# Replace with real calibration fitted from your standards.
# concentration = a * darkness + b
CAL_A = 120.0
CAL_B = 0.0

def estimate_darkness(img_bgr: np.ndarray) -> float:
    # 1) grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2) simple ROI strategy: use center crop (replace with your real ROI)
    h, w = gray.shape
    crop = gray[int(0.35*h):int(0.65*h), int(0.35*w):int(0.65*w)]

    # 3) mean intensity -> darkness
    mean_intensity = float(np.mean(crop))  # 0..255
    darkness = 1.0 - (mean_intensity / 255.0)  # 0..1 (higher = darker)
    return float(np.clip(darkness, 0.0, 1.0))

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
        "darkness_score": darkness,
        "trypsin_estimate": trypsin,
        "units": "YOUR_UNITS_HERE",
        "notes": "Calibration is placeholder until you fit to your standards."
    }