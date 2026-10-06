"""
Face-expression recognition with an exported ONNX ensemble.

Self-contained: needs only numpy, opencv-python and onnxruntime (or onnxruntime-gpu).
The same FaceCropper is used to build the training crops (fer_lib.py imports it from here),
so live crops look exactly like the training images.

Usage
-----
    python fer_infer_onnx.py --model-dir artifacts/fer_kfold/export --image photo.jpg
    python fer_infer_onnx.py --model-dir artifacts/fer_kfold/export --folder faces/ --out-csv preds.csv
    python fer_infer_onnx.py --model-dir artifacts/fer_kfold/export --webcam

In code
-------
    from fer_infer_onnx import FaceExpressionRecognizer
    rec = FaceExpressionRecognizer("artifacts/fer_kfold/export")
    results = rec.analyse(cv2.imread("photo.jpg"))   # list of dicts (box, label, confidence, probs)
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------------------
# Image I/O (works with non-ASCII Windows paths)
# --------------------------------------------------------------------------------------
def read_bgr(path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except Exception:
        return None


# --------------------------------------------------------------------------------------
# Face detection + cropping (YuNet)
# --------------------------------------------------------------------------------------
class FaceCropper:
    """YuNet face detector + square face crop with a margin.

    margin: extra space around the detected box on every side, as a fraction of the box size.
            0.2 -> the crop side is 1.4x the larger side of the face box.
    """

    def __init__(self, model_path, margin: float = 0.2, score_threshold: float = 0.6,
                 nms_threshold: float = 0.3, top_k: int = 50, max_side: int = 1280, min_side: int = 320):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"YuNet model not found: {model_path}\n"
                "Download face_detection_yunet_2023mar.onnx from "
                "https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet")
        self.margin = float(margin)
        self.max_side = int(max_side)
        self.min_side = int(min_side)
        self.detector = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320),
                                                  score_threshold, nms_threshold, top_k)

    def detect(self, img_bgr: np.ndarray) -> np.ndarray:
        """Return an (N, 5) array of [x, y, w, h, score] in original pixel coordinates,
        largest face first."""
        h, w = img_bgr.shape[:2]
        # Very large photos are slow and very small ones are missed -> detect on a rescaled copy
        scale = 1.0
        if max(h, w) > self.max_side:
            scale = self.max_side / max(h, w)
        elif min(h, w) < self.min_side:
            scale = self.min_side / min(h, w)
        img = img_bgr if scale == 1.0 else cv2.resize(
            img_bgr, (max(1, round(w * scale)), max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        self.detector.setInputSize((img.shape[1], img.shape[0]))
        _, faces = self.detector.detect(img)
        if faces is None or len(faces) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        boxes = np.concatenate([faces[:, :4] / scale, faces[:, 14:15]], axis=1).astype(np.float32)
        order = np.argsort(-(boxes[:, 2] * boxes[:, 3]))
        return boxes[order]

    def crop(self, img_bgr: np.ndarray, box) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        return square_crop(img_bgr, box, self.margin)

    def largest_face(self, img_bgr: np.ndarray, assume_crop_below: int = 0):
        """(crop, found). Images whose short side is below `assume_crop_below` px (e.g. 48x48
        FER-style faces) are treated as face crops already. When no face is found, the centre
        square of the image is returned."""
        h, w = img_bgr.shape[:2]
        if min(h, w) >= assume_crop_below:
            boxes = self.detect(img_bgr)
            if len(boxes):
                crop, _ = self.crop(img_bgr, boxes[0])
                if crop.size:
                    return crop, True
        return center_square(img_bgr), False


def center_square(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return img_bgr[y0:y0 + s, x0:x0 + s]


def square_crop(img_bgr: np.ndarray, box, margin: float = 0.2):
    """Square crop around `box` (x, y, w, h, ...) with `margin` (fraction of the box size added on
    every side). Image borders are padded by replicating edge pixels so the face stays centred.
    Returns (crop, (x1, y1, x2, y2)) where the rectangle is clipped to the image."""
    x, y, bw, bh = [float(v) for v in box[:4]]
    cx, cy = x + bw / 2.0, y + bh / 2.0
    side = max(bw, bh) * (1.0 + 2.0 * margin)
    x1, y1 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x2, y2 = int(round(cx + side / 2)), int(round(cy + side / 2))
    h, w = img_bgr.shape[:2]
    pad_l, pad_t = max(0, -x1), max(0, -y1)
    pad_r, pad_b = max(0, x2 - w), max(0, y2 - h)
    crop = img_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if crop.size and (pad_l or pad_t or pad_r or pad_b):
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
    rect = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
    return crop, rect


# --------------------------------------------------------------------------------------
# Classifier preprocessing (must match the training/validation transform)
# --------------------------------------------------------------------------------------
def preprocess_faces(crops_bgr, size: int) -> np.ndarray:
    """BGR face crops -> float32 NCHW RGB in [0, 1]: short side resized to `size`, centre crop.
    ImageNet normalisation and softmax are inside the exported ONNX model."""
    out = np.empty((len(crops_bgr), 3, size, size), dtype=np.float32)
    for i, img in enumerate(crops_bgr):
        h, w = img.shape[:2]
        s = size / min(h, w)
        nw, nh = max(size, round(w * s)), max(size, round(h * s))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        y0, x0 = (nh - size) // 2, (nw - size) // 2
        img = img[y0:y0 + size, x0:x0 + size]
        out[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1) / 255.0
    return out


# --------------------------------------------------------------------------------------
# Recogniser
# --------------------------------------------------------------------------------------
class FaceExpressionRecognizer:
    """Loads `<model_dir>/deployment_metadata.json` + the ONNX model it names."""

    def __init__(self, model_dir, providers=None, detector_path=None):
        import onnxruntime as ort

        self.model_dir = Path(model_dir)
        self.meta = json.loads((self.model_dir / "deployment_metadata.json").read_text(encoding="utf-8"))
        self.classes = list(self.meta["classes"])
        self.image_size = int(self.meta["image_size"])
        self.conf_threshold = float(self.meta.get("confidence_threshold", 0.5))
        self.assume_crop_below = int(self.meta.get("assume_crop_below", 0))
        if providers is None:
            available = ort.get_available_providers()
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(str(self.model_dir / self.meta.get("onnx_file", "model.onnx")),
                                            providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        det = Path(detector_path) if detector_path else self.model_dir / self.meta.get(
            "face_detector", "face_detection_yunet_2023mar.onnx")
        self.cropper = FaceCropper(det, margin=float(self.meta.get("face_margin", 0.2))) if det.exists() else None

    def classify_faces(self, crops_bgr, batch_size: int = 64) -> np.ndarray:
        """List of BGR face crops -> (N, num_classes) probabilities in `self.classes` order."""
        if len(crops_bgr) == 0:
            return np.zeros((0, len(self.classes)), dtype=np.float32)
        outs = []
        for i in range(0, len(crops_bgr), batch_size):
            x = preprocess_faces(crops_bgr[i:i + batch_size], self.image_size)
            outs.append(self.session.run(None, {self.input_name: x})[0])
        return np.concatenate(outs).astype(np.float32)

    def analyse(self, img_bgr: np.ndarray, max_faces: int = 10):
        """Detect every face in a full image and classify it."""
        if self.cropper is None:
            raise RuntimeError("Face detector not found next to the model; pass detector_path=...")
        crops, rects = [], []
        h, w = img_bgr.shape[:2]
        boxes = [] if min(h, w) < self.assume_crop_below else self.cropper.detect(img_bgr)[:max_faces]
        if len(boxes) == 0 and min(h, w) < max(self.assume_crop_below, 1):
            crops, rects = [center_square(img_bgr)], [(0, 0, w, h)]   # small image = a face crop already
        for box in boxes:
            crop, rect = self.cropper.crop(img_bgr, box)
            if crop.size:
                crops.append(crop)
                rects.append(rect)
        probs = self.classify_faces(crops)
        results = []
        for rect, p in zip(rects, probs):
            cid = int(p.argmax())
            results.append({"box": rect, "label": self.classes[cid], "confidence": float(p[cid]),
                            "uncertain": bool(p[cid] < self.conf_threshold),
                            "probs": dict(zip(self.classes, map(float, p)))})
        return results


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def _draw(frame, rect, text, ok):
    color = (0, 200, 0) if ok else (0, 200, 255)
    x1, y1, x2, y2 = rect
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, text, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def run_webcam(rec: FaceExpressionRecognizer, camera: int = 0, smoothing: float = 0.6):
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera}")
    smoothed, t_prev = {}, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        new = {}
        for r in rec.analyse(frame):
            x1, y1, x2, y2 = r["box"]
            key = ((x1 + x2) // 80, (y1 + y2) // 80)          # rough identity by position
            p = np.array([r["probs"][c] for c in rec.classes])
            if smoothing > 0 and key in smoothed:
                p = smoothing * smoothed[key] + (1 - smoothing) * p
            new[key] = p
            cid = int(p.argmax())
            conf = float(p[cid])
            ok_conf = conf >= rec.conf_threshold
            text = f"{rec.classes[cid]} {conf * 100:.0f}%" if ok_conf else f"Uncertain ({rec.classes[cid]}?)"
            _draw(frame, r["box"], text, ok_conf)
        smoothed = new
        t = time.perf_counter()
        cv2.putText(frame, f"{1 / max(t - t_prev, 1e-6):.0f} FPS", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        t_prev = t
        cv2.imshow("Face expression (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--image", help="full photo: detect faces and classify each")
    ap.add_argument("--folder", help="folder of images -> CSV (largest face per image)")
    ap.add_argument("--out-csv", default="face_expression_predictions.csv")
    ap.add_argument("--output", help="annotated image for --image (default: <name>_pred.jpg in the current folder)")
    ap.add_argument("--crops", action="store_true", help="inputs are already face crops (skip detection)")
    ap.add_argument("--webcam", action="store_true")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--smoothing", type=float, default=0.6)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    rec = FaceExpressionRecognizer(args.model_dir, providers=["CPUExecutionProvider"] if args.cpu else None)
    print("Providers:", rec.session.get_providers(), "| classes:", rec.classes)

    if args.image:
        img = read_bgr(args.image)
        if img is None:
            raise SystemExit(f"Cannot read {args.image}")
        if args.crops:
            p = rec.classify_faces([img])[0]
            print({c: round(float(v), 4) for c, v in zip(rec.classes, p)})
        else:
            res = rec.analyse(img)
            if not res:
                print("No face found.")
            for r in res:
                print(f"{r['label']:>9s} {r['confidence'] * 100:5.1f}%  box={r['box']}"
                      + ("  (uncertain)" if r["uncertain"] else ""))
                _draw(img, r["box"], f"{r['label']} {r['confidence'] * 100:.0f}%", not r["uncertain"])
            out = Path(args.output) if args.output else Path.cwd() / (Path(args.image).stem + "_pred.jpg")
            cv2.imencode(".jpg", img)[1].tofile(str(out))
            print("Saved", out)

    if args.folder:
        paths = sorted(p for p in Path(args.folder).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        rows = []
        for p in paths:
            img = read_bgr(p)
            if img is None:
                continue
            found = True
            if not args.crops:
                img, found = rec.cropper.largest_face(img, rec.assume_crop_below)
            prob = rec.classify_faces([img])[0]
            cid = int(prob.argmax())
            rows.append({"image": str(p), "expression": rec.classes[cid], "confidence": float(prob[cid]),
                         "uncertain": bool(prob[cid] < rec.conf_threshold), "face_found": found,
                         **{f"p_{c}": float(v) for c, v in zip(rec.classes, prob)}})
        if rows:
            with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        print(f"Predicted {len(rows)} images -> {args.out_csv}")

    if args.webcam:
        run_webcam(rec, args.camera, args.smoothing)


if __name__ == "__main__":
    main()
