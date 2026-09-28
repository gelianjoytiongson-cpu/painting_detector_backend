import base64
import os

import cv2
import numpy as np
from flask import Flask, jsonify, request

from get_painting_reference import (
    detect_painting_box,
    crop_painting,
    extract_features,
    fit_to_max_side,
    step_convert_to_grayscale,
    PAINTING_SIZE,
    SCALAR_KEYS,
    QUADRANT_KEYS,
)

REFERENCE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "painting_reference.npy")
MIN_REL_TOLERANCE = 0.15
MATCH_THRESHOLD = 3.0

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15 MB upload limit

# Optional API key: set API_KEY in Render's environment variables to enable.
API_KEY = os.environ.get("API_KEY")


@app.before_request
def check_key():
    if API_KEY and request.path != "/health" and request.headers.get("X-API-Key") != API_KEY:
        return jsonify(error="Unauthorized"), 401


if not os.path.isfile(REFERENCE_FILE):
    raise IOError(f"Missing {REFERENCE_FILE}. Run get_painting_reference.py and copy the file here.")
REFERENCE = np.load(REFERENCE_FILE, allow_pickle=True).item()


def deviation(live_value, ref_mean, ref_std):
    spread = max(ref_std, MIN_REL_TOLERANCE * abs(ref_mean))
    return abs(live_value - ref_mean) / spread


def compare_to_reference(features, reference):
    mean, std = reference["mean"], reference["std"]
    per_feature = {}
    for key in SCALAR_KEYS:
        per_feature[key] = deviation(features[key], mean[key], std[key])
    for q in QUADRANT_KEYS:
        per_feature[f"quadrant_{q}"] = deviation(
            features["quadrant_densities"][q],
            mean["quadrant_densities"][q],
            std["quadrant_densities"][q],
        )
    return float(np.mean(list(per_feature.values()))), per_feature


def parse_manual_box(text, img_w, img_h):
    """Optional 'x,y,w,h' as fractions (0-1) of the image. Returns pixel box or None."""
    try:
        x, y, w, h = [float(v) for v in text.split(",")]
    except (ValueError, AttributeError):
        return None
    if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1):
        return None
    box = (int(x * img_w), int(y * img_h), int(w * img_w), int(h * img_h))
    return box if box[2] > 10 and box[3] > 10 else None


@app.get("/health")
def health():
    return jsonify(status="ok", reference_samples=REFERENCE["num_samples"])


@app.post("/verify")
def verify():
    file = request.files.get("image")
    if file is None:
        return jsonify(error="No 'image' file in the request."), 400

    data = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)  # applies the phone's EXIF rotation
    if img is None:
        return jsonify(error="Could not read the image."), 400
    img = fit_to_max_side(img)

    ih, iw = img.shape[:2]
    manual = parse_manual_box(request.form.get("box"), iw, ih)

    if manual is not None:
        painting_bgr = crop_painting(img, manual, None)
        mode = "manual"
    else:
        box, quad = detect_painting_box(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if quad is None:
            return jsonify(found=False,
                           message="No painting detected. Retake the photo straight-on "
                                   "with the whole painting in frame."), 200
        painting_bgr = crop_painting(img, box, quad)
        mode = "auto"

    # No mirroring: the reference and this photo both come from a phone camera.
    features, _ = extract_features(step_convert_to_grayscale(painting_bgr))
    score, breakdown = compare_to_reference(features, REFERENCE)

    ok, jpg = cv2.imencode(".jpg", cv2.resize(painting_bgr, (PAINTING_SIZE[0] // 2, PAINTING_SIZE[1] // 2)))
    preview = base64.b64encode(jpg.tobytes()).decode() if ok else None

    return jsonify(
        found=True,
        mode=mode,
        is_match=score <= MATCH_THRESHOLD,
        score=round(score, 3),
        threshold=MATCH_THRESHOLD,
        breakdown={k: round(v, 3) for k, v in breakdown.items()},
        cropped_preview=preview,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
