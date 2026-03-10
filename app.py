from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2

app = FastAPI(title="Trypsin Estimator")

# Friendly home route so "/" isn't a 404
@app.get("/")
def home():
    return {
        "message": "Trypsin Estimator API is running.",
        "how_to_use": "POST an image to /predict, or open /docs for the upload UI.",
        "endpoints": ["/predict", "/docs", "/redoc"],
    }

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