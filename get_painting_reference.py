import os
import sys
import time

import cv2
import numpy as np

PAINTING_SIZE = (600, 800)       # (width, height), portrait
MONTAGE_CELL = (150, 200)        # small cells so the montage stays a reasonable size
PHOTO_FOLDER = "photos"          # put your phone photos here
PIPELINE_DEMO_FOLDER = "outputs"
MAX_SIDE = 1000                  # phone photos are huge, so shrink them for detection and display
MIN_AREA_FRACTION = 0.05
MAX_AREA_FRACTION = 0.95
EXPECTED_RATIO = 0.75   # short side / long side of your painting (600x800 -> 0.75)
RATIO_TOL = 0.20        # how far the detected ratio may differ from that
BORDER_MARGIN = 0.02    # reject regions touching the frame edge (fraction of frame size)
INSET = 0.03            # trim this fraction on each side to drop frame/background bleed
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

SCALAR_KEYS = ["mean_intensity", "std_intensity", "edge_density"]
QUADRANT_KEYS = ["top_left", "top_right", "bottom_left", "bottom_right"]


def fit_to_max_side(img, max_side=MAX_SIDE):
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


# ---- Painting detection and cropping ----
def order_points(pts):
    """Orders 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def _quad_sides(quad):
    tl, tr, br, bl = quad
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    return w, h


def shrink_quad(poly, inset=INSET):
    """Ordered corners pulled in toward the center. This is exactly the area that gets cropped."""
    quad = order_points(poly)
    center = quad.mean(axis=0)
    return center + (quad - center) * (1 - 2 * inset)


def detect_painting_box(gray_full):
    """Finds the painting as a 4-corner shape. Returns ((x, y, w, h), ordered_corners) or (None, None)."""
    fh, fw = gray_full.shape
    frame_area = fh * fw
    margin = BORDER_MARGIN * min(fh, fw)

    blurred = cv2.GaussianBlur(gray_full, (7, 7), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.Canny(blurred, 30, 100)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best_quad, best_area = None, 0
    for c in contours:
        area = cv2.contourArea(c)
        if not (MIN_AREA_FRACTION * frame_area <= area <= MAX_AREA_FRACTION * frame_area):
            continue

        hull = cv2.convexHull(c)
        approx = cv2.approxPolyDP(hull, 0.03 * cv2.arcLength(hull, True), True)
        if len(approx) != 4:
            continue

        quad = order_points(approx)
        quad_area = cv2.contourArea(quad)
        if quad_area <= 0 or cv2.contourArea(hull) / quad_area < 0.90:
            continue  # not really a clean rectangle

        # background usually touches the frame edge; the painting shouldn't
        if (quad[:, 0].min() < margin or quad[:, 1].min() < margin or
                quad[:, 0].max() > fw - margin or quad[:, 1].max() > fh - margin):
            continue

        w, h = _quad_sides(quad)
        ratio = min(w, h) / max(w, h)
        if abs(ratio - EXPECTED_RATIO) > RATIO_TOL:
            continue  # wrong proportions: shelf, window, table edge, etc.

        if quad_area > best_area:
            best_quad, best_area = quad, quad_area

    if best_quad is None:
        return None, None
    box = cv2.boundingRect(best_quad.astype(np.int32).reshape(-1, 1, 2))
    return box, best_quad


def crop_painting(img, box, poly, size=PAINTING_SIZE, inset=INSET):
    """Crops only what's inside the painting's corners (perspective-corrected).
    Falls back to a plain rectangle crop when there are no corners (manual selection)."""
    w_out, h_out = size

    if poly is None:
        ih, iw = img.shape[:2]
        x, y, w, h = box
        x, y = max(int(x), 0), max(int(y), 0)
        w, h = min(int(w), iw - x), min(int(h), ih - y)
        return cv2.resize(img[y:y + h, x:x + w], size)

    quad = shrink_quad(poly, inset)
    w, h = _quad_sides(quad)
    if w > h:  # landscape -> rotate to match the portrait reference
        quad = np.roll(quad, -1, axis=0)

    dst = np.float32([[0, 0], [w_out - 1, 0], [w_out - 1, h_out - 1], [0, h_out - 1]])
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (w_out, h_out))


# ---- Pipeline steps (teacher's diagram) ----
def step_convert_to_grayscale(painting_bgr):
    return cv2.cvtColor(painting_bgr, cv2.COLOR_BGR2GRAY)


def step_noise_reduction(gray):
    return cv2.GaussianBlur(gray, (5, 5), 0)


def step_calculate_gradient(blurred):
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    return cv2.convertScaleAbs(cv2.magnitude(sobel_x, sobel_y))


def step_apply_edge_detection(blurred):
    return cv2.Canny(blurred, 50, 150)


# ---- Features ----
def extract_features(painting_gray):
    blurred = step_noise_reduction(painting_gray)
    edges = step_apply_edge_detection(blurred)

    h, w = edges.shape
    quadrants = {
        "top_left": edges[:h // 2, :w // 2],
        "top_right": edges[:h // 2, w // 2:],
        "bottom_left": edges[h // 2:, :w // 2],
        "bottom_right": edges[h // 2:, w // 2:],
    }

    features = {
        "mean_intensity": float(np.mean(blurred)),
        "std_intensity": float(np.std(blurred)),
        "edge_density": float(np.sum(edges == 255) / edges.size),
        "quadrant_densities": {
            name: float(np.sum(q == 255) / q.size) for name, q in quadrants.items()
        },
    }
    return features, edges


def average_features(samples):
    mean_out, std_out = {}, {}

    for key in SCALAR_KEYS:
        values = [s[key] for s in samples]
        mean_out[key], std_out[key] = float(np.mean(values)), float(np.std(values))

    mean_out["quadrant_densities"], std_out["quadrant_densities"] = {}, {}
    for q in QUADRANT_KEYS:
        values = [s["quadrant_densities"][q] for s in samples]
        mean_out["quadrant_densities"][q] = float(np.mean(values))
        std_out["quadrant_densities"][q] = float(np.std(values))

    return mean_out, std_out


# ---- Report images ----
def make_montage(labeled_images, cols=4):
    cells = []
    for label, img in labeled_images:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cell = cv2.resize(img, MONTAGE_CELL)
        cv2.putText(cell, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 255, 255), 1, cv2.LINE_AA)
        cells.append(cell)

    while len(cells) % cols:  # pad the last row with blank cells
        cells.append(np.zeros_like(cells[0]))

    return np.vstack([np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)])


def save_pipeline_demo(painting_bgr, sample_number, traced_photo=None):
    folder = os.path.join(PIPELINE_DEMO_FOLDER, f"sample_{sample_number}")
    os.makedirs(folder, exist_ok=True)

    gray = step_convert_to_grayscale(painting_bgr)
    blurred = step_noise_reduction(gray)
    gradient = step_calculate_gradient(blurred)
    edges = step_apply_edge_detection(blurred)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_view = np.zeros((*edges.shape, 3), dtype=np.uint8)
    cv2.drawContours(contour_view, contours, -1, (0, 255, 0), 1)

    steps = [
        ("Original", painting_bgr),
        ("Grayscale", gray),
        ("Noise Reduction", blurred),
        ("Image Gradient", gradient),
        ("Edge Detection", edges),
        ("Edge Image", edges),
        ("Contours", contour_view),
    ]

    if traced_photo is not None:
        cv2.imwrite(os.path.join(folder, "00_photo_with_trace.png"), traced_photo)

    for i, (label, img) in enumerate(steps, start=1):
        name = f"{i:02d}_{label.lower().replace(' ', '_')}.png"
        cv2.imwrite(os.path.join(folder, name), img)

    montage = make_montage([(f"{i} {label}", img) for i, (label, img) in enumerate(steps, start=1)])
    cv2.imwrite(os.path.join(folder, "00_all_steps_montage.png"), montage)
    return montage


def list_photos():
    if not os.path.isdir(PHOTO_FOLDER):
        return []
    names = sorted(n for n in os.listdir(PHOTO_FOLDER) if n.lower().endswith(IMAGE_EXTENSIONS))
    return [os.path.join(PHOTO_FOLDER, n) for n in names]


def draw_trace(img, box, poly):
    """Draws the crop area on a copy of the photo: yellow = detected corners, green = what gets cropped."""
    trace = img.copy()
    if poly is not None:
        cv2.polylines(trace, [order_points(poly).astype(np.int32).reshape(-1, 1, 2)],
                      True, (0, 255, 255), 2)
        cv2.polylines(trace, [shrink_quad(poly).astype(np.int32).reshape(-1, 1, 2)],
                      True, (0, 255, 0), 2)
    elif box is not None:
        x, y, w, h = box
        cv2.rectangle(trace, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return trace


def capture_photos_from_phone():
    """Live view from the phone's back camera. 'c' saves a full-resolution photo, 'q' when done."""
    from camera_source import open_camera

    os.makedirs(PHOTO_FOLDER, exist_ok=True)
    cap = open_camera()
    print("Point the phone's back camera at the painting.")
    print("Press 'c' to save a photo, 'q' when you have enough.")
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lost the camera stream.")
            break

        display = fit_to_max_side(frame)
        box, poly = detect_painting_box(cv2.cvtColor(display, cv2.COLOR_BGR2GRAY))
        display = draw_trace(display, box, poly)
        status = "painting found" if box is not None else "no painting found"
        cv2.putText(display, f"Saved: {saved}  (c=save, q=done)  [{status}]", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.imshow("Phone camera capture", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            path = os.path.join(PHOTO_FOLDER, f"phone_{int(time.time() * 1000)}.jpg")
            cv2.imwrite(path, frame)
            saved += 1
            print(f"Saved {path}")

    cap.release()
    cv2.destroyAllWindows()


def main():
    os.makedirs(PIPELINE_DEMO_FOLDER, exist_ok=True)

    if "--camera" in sys.argv:
        capture_photos_from_phone()

    paths = list_photos()
    if not paths:
        raise FileNotFoundError(
            f"No photos found. Create a folder named '{PHOTO_FOLDER}' and put your phone photos in it.")

    print(f"Found {len(paths)} photo(s).")
    print("Enter/Space = accept crop, r = draw rectangle manually, n = skip, q = quit")

    window = "Painting Reference Photos"
    samples, montages = [], []
    quit_all = False

    for path in paths:
        if quit_all:
            break

        img = cv2.imread(path)  # OpenCV applies the phone's EXIF rotation automatically
        if img is None:
            print(f"Could not read {path}, skipping.")
            continue
        img = fit_to_max_side(img)

        box, poly = detect_painting_box(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

        while True:
            trace = draw_trace(img, box, poly)   # clean trace, saved with the sample
            display = trace.copy()

            if box is not None:
                status = "painting found - Enter=accept, r=redraw, n=skip"
            else:
                status = "no painting found - press r to draw it, n=skip"

            cv2.putText(display, f"{os.path.basename(path)}: {status}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.imshow(window, display)

            key = cv2.waitKey(0) & 0xFF

            if key == ord('q'):
                quit_all = True
                break
            if key == ord('n'):
                break
            if key == ord('r'):
                roi = cv2.selectROI(window, img, showCrosshair=False)
                if roi[2] > 10 and roi[3] > 10:
                    box, poly = tuple(int(v) for v in roi), None
                continue
            if key in (13, 32) and box is not None:
                painting_bgr = crop_painting(img, box, poly)   # same crop as verify_painting.py
                features, _ = extract_features(step_convert_to_grayscale(painting_bgr))
                samples.append(features)
                montages.append(save_pipeline_demo(painting_bgr, len(samples), traced_photo=trace))
                cv2.imshow("Cropped painting (what is measured)", painting_bgr)
                print(f"Sample {len(samples)} ({os.path.basename(path)}): "
                      f"mean_intensity={features['mean_intensity']:.2f}  "
                      f"edge_density={features['edge_density']:.4f}")
                break

    cv2.destroyAllWindows()

    if not samples:
        print("No samples accepted. Nothing saved.")
        return

    mean_features, std_features = average_features(samples)
    np.save("painting_reference.npy",
            {"num_samples": len(samples), "mean": mean_features, "std": std_features},
            allow_pickle=True)

    print("\nReference saved to painting_reference.npy")
    print("Averaged values (mean +/- std):")
    for key in SCALAR_KEYS:
        print(f"  {key}: {mean_features[key]:.4f} +/- {std_features[key]:.4f}")
    for q in QUADRANT_KEYS:
        m = mean_features["quadrant_densities"][q]
        s = std_features["quadrant_densities"][q]
        print(f"  {q}: {m:.4f} +/- {s:.4f}")

    cv2.imwrite(os.path.join(PIPELINE_DEMO_FOLDER, "all_samples_overview.png"), np.vstack(montages))
    for i, m in enumerate(montages, start=1):
        cv2.imshow(f"Sample {i} - pipeline steps", m)
    print("\nPress any key in an image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
