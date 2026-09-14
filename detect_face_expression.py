#!/usr/bin/env python
"""
Standalone Python script to detect faces and classify emotions from an image file
using OpenCV YuNet face detection and ONNX Runtime (face_emotion_model.onnx).

Usage:
    python detect_face_expression.py path/to/your/image.jpg
    python detect_face_expression.py path/to/your/image.jpg --output result.jpg
"""

import os
import sys
import json
import urllib.request
import argparse
import numpy as np
import cv2
import onnxruntime as ort

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'face_expression_model')
YUNET_PATH = os.path.join(MODEL_DIR, 'face_detection_yunet_2023mar.onnx')
EMOTION_ONNX_PATH = os.path.join(MODEL_DIR, 'face_emotion_model.onnx')
CLASSES_PATH = os.path.join(MODEL_DIR, 'class_names.json')

# Load Emotion Classes
try:
    with open(CLASSES_PATH, 'r') as f:
        CLASS_NAMES = json.load(f)
except Exception:
    CLASS_NAMES = ["Angry", "Fear", "Happy", "Sad", "Suprise"]

# Color palette for 5 emotions (BGR)
EMOTION_COLORS = {
    'Angry': (78, 63, 244),      # Crimson Rose
    'Fear': (246, 92, 139),       # Purple
    'Happy': (129, 185, 16),      # Mint Emerald
    'Sad': (212, 182, 6),         # Cyan Blue
    'Suprise': (11, 158, 245),   # Electric Amber
    'Surprise': (11, 158, 245),  # Electric Amber
}

EMOTION_EMOJIS = {
    'Angry': '😠', 'Fear': '😨', 'Happy': '😃', 'Sad': '😢', 'Suprise': '😲', 'Surprise': '😲'
}


class FaceEmotionAnalyzer:
    """Detects faces and predicts facial emotions from images."""

    def __init__(self, yunet_path=YUNET_PATH, emotion_model_path=EMOTION_ONNX_PATH):
        # 1. Ensure YuNet Face Detector is available
        if not os.path.exists(yunet_path):
            print(f"[*] Downloading lightweight YuNet face detector to {yunet_path}...")
            url = 'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
            os.makedirs(os.path.dirname(yunet_path), exist_ok=True)
            urllib.request.urlretrieve(url, yunet_path)

        self.detector = cv2.FaceDetectorYN_create(
            model=yunet_path,
            config='',
            input_size=(320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000
        )

        # 2. Initialize ONNX Emotion Model
        if not os.path.exists(emotion_model_path):
            raise FileNotFoundError(f"Emotion model not found at {emotion_model_path}")
        
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(emotion_model_path, sess_options=opts)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [1, 3, 48, 48]
        print(f"[OK] Initialized Emotion Model: {os.path.basename(emotion_model_path)}")

    def detect_faces(self, image_bgr):
        """Detect faces in BGR image using YuNet. Falls back to whole image if no face detected."""
        h, w = image_bgr.shape[:2]
        self.detector.setInputSize((w, h))
        _, detected = self.detector.detect(image_bgr)
        
        faces = []
        if detected is not None:
            for det in detected:
                box = det[0:4].astype(int)
                score = float(det[-1])
                bx = max(0, int(box[0]))
                by = max(0, int(box[1]))
                bw = min(w - bx, int(box[2]))
                bh = min(h - by, int(box[3]))
                if bw > 15 and bh > 15:
                    landmarks = det[4:14].reshape((5, 2)).astype(int).tolist() if len(det) >= 14 else []
                    faces.append({
                        'box': [bx, by, bw, bh],
                        'confidence': score,
                        'landmarks': landmarks,
                        'is_fallback': False
                    })

        # Fallback to whole frame if it's already a cropped face image
        if not faces:
            faces.append({
                'box': [0, 0, w, h],
                'confidence': 0.90,
                'landmarks': [],
                'is_fallback': True
            })

        return faces

    def preprocess_face(self, image_bgr, box, padding=0.10):
        """Crop and resize face to 48x48 RGB in NCHW format."""
        h_img, w_img = image_bgr.shape[:2]
        x, y, w, h = box

        if w == w_img and h == h_img:
            crop = image_bgr
        else:
            pad_x = int(w * padding)
            pad_y = int(h * padding)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w_img, x + w + pad_x)
            y2 = min(h_img, y + h + pad_y)
            crop = image_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                crop = image_bgr

        # Resize to 48x48 and convert BGR -> RGB
        resized = cv2.resize(crop, (48, 48), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0.0, 1.0] and format as (1, 3, 48, 48)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        return crop, tensor

    def predict_emotion(self, tensor):
        """Run ONNX model and compute softmax probabilities."""
        outputs = self.session.run(None, {self.input_name: tensor})
        raw = outputs[0][0]  # Raw logits

        num_classes = len(CLASS_NAMES)
        logits = raw[:num_classes].astype(np.float64)

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)

        top_idx = int(np.argmax(probabilities))
        primary = CLASS_NAMES[top_idx]
        confidence = float(probabilities[top_idx])

        prob_dict = {
            CLASS_NAMES[i]: round(float(probabilities[i]) * 100, 2)
            for i in range(num_classes)
        }
        return primary, confidence, prob_dict

    def analyze_image(self, image_path, output_path=None):
        """Complete pipeline: reads image, detects faces, predicts emotions, draws results."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not decode image at: {image_path}")

        faces = self.detect_faces(img)
        print(f"\n" + "=" * 60)
        print(f" Image Analysis: {os.path.basename(image_path)}")
        print(f" Resolution: {img.shape[1]}x{img.shape[0]} | Faces Detected: {len(faces)}")
        print("=" * 60)

        annotated = img.copy()
        results = []

        for idx, face in enumerate(faces):
            box = face['box']
            crop, tensor = self.preprocess_face(img, box)
            primary, conf, probs = self.predict_emotion(tensor)

            color = EMOTION_COLORS.get(primary, (255, 100, 0))
            face_type = "Full Image (Auto-crop)" if face.get('is_fallback') else f"Detected Face #{idx + 1}"
            
            print(f"\n[*] {face_type} (Box: x={box[0]}, y={box[1]}, w={box[2]}, h={box[3]}):")
            print(f"    Primary Emotion: {primary} ({conf * 100:.1f}% confidence)")
            print(f"    Probability Breakdown:")
            for emo, p in probs.items():
                bar = "#" * int(p / 4)
                print(f"      - {emo:8s} {p:5.1f}% | {bar}")

            # Draw sleek bounding box on image
            x, y, w, h = box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Draw corner brackets
            corner_len = min(20, w // 4, h // 4)
            cv2.line(annotated, (x, y), (x + corner_len, y), (255, 255, 255), 3)
            cv2.line(annotated, (x, y), (x, y + corner_len), (255, 255, 255), 3)
            cv2.line(annotated, (x + w, y), (x + w - corner_len, y), (255, 255, 255), 3)
            cv2.line(annotated, (x + w, y), (x + w, y + corner_len), (255, 255, 255), 3)
            cv2.line(annotated, (x, y + h), (x + corner_len, y + h), (255, 255, 255), 3)
            cv2.line(annotated, (x, y + h), (x, y + h - corner_len), (255, 255, 255), 3)
            cv2.line(annotated, (x + w, y + h), (x + w - corner_len, y + h), (255, 255, 255), 3)
            cv2.line(annotated, (x + w, y + h), (x + w, y + h - corner_len), (255, 255, 255), 3)

            # Draw label banner
            label = f"{primary} {round(conf * 100)}%"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            badge_y = max(y - 8, lh + 8)
            cv2.rectangle(annotated, (x, badge_y - lh - 6), (x + lw + 10, badge_y + 4), color, -1)
            cv2.putText(annotated, label, (x + 5, badge_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            results.append({
                'face_id': idx + 1,
                'box': box,
                'primary_emotion': primary,
                'confidence': round(conf * 100, 2),
                'probabilities': probs
            })

        # Save annotated image if requested
        if output_path:
            cv2.imwrite(output_path, annotated)
            print(f"\n[OK] Annotated image saved to: {output_path}")

        return {
            'faces_found': len(faces),
            'results': results,
            'annotated_image': annotated
        }


def main():
    parser = argparse.ArgumentParser(description="Identify faces and classify emotions from an image.")
    parser.add_argument('image', type=str, help="Path to input image file (JPG, PNG, WebP)")
    parser.add_argument('--output', '-o', type=str, default='output_detected.jpg',
                        help="Path to save the annotated output image (default: output_detected.jpg)")
    args = parser.parse_args()

    analyzer = FaceEmotionAnalyzer()
    analyzer.analyze_image(args.image, output_path=args.output)


if __name__ == '__main__':
    main()
