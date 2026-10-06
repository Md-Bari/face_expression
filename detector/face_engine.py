"""
Face Detection and Emotion Recognition Engine.
Integrates OpenCV YuNet Face Detector with ONNX facial-expression models.

Two kinds of models are supported:
* v2 models trained with training/face_expression_kfold.ipynb: described by
  face_expression_model/deployment_metadata.json (input size, classes, face margin; the ONNX graph
  outputs probabilities). When present, this model is the default.
* legacy models (face_emotion_model.onnx etc.): 48x48 RGB input, logits output, class_names.json.
"""

import os
import io
import json
import base64
import time
import numpy as np
import cv2
from PIL import Image
from django.conf import settings
import onnxruntime as ort

from training.fer_infer_onnx import center_square, preprocess_faces, square_crop

# Legacy class names (face_emotion_model.onnx and other 48x48 models)
CLASSES_PATH = settings.MODEL_DIR / 'class_names.json'
try:
    with open(CLASSES_PATH, 'r') as f:
        LEGACY_CLASS_NAMES = json.load(f)
except Exception:
    LEGACY_CLASS_NAMES = ["Angry", "Fear", "Happy", "Sad", "Suprise"]

# v2 model trained by training/face_expression_kfold.ipynb (preferred when present)
METADATA_PATH = settings.MODEL_DIR / 'deployment_metadata.json'
V2_META = None
if os.path.exists(METADATA_PATH):
    try:
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            V2_META = json.load(f)
        if not os.path.exists(settings.MODEL_DIR / V2_META.get('onnx_file', '')):
            print(f"[FaceEngine] {METADATA_PATH.name} found but {V2_META.get('onnx_file')} is missing - ignored")
            V2_META = None
    except Exception as e:
        print(f"[FaceEngine] Could not read {METADATA_PATH}: {e}")
        V2_META = None

PRIMARY_MODEL_FILE = V2_META['onnx_file'] if V2_META else 'face_emotion_model.onnx'
# Class names of the default model (used by the views for statistics and the UI)
CLASS_NAMES = list(V2_META['classes']) if V2_META else LEGACY_CLASS_NAMES

# Color palette for the 5 emotions (BGR for OpenCV, Hex for UI)
EMOTION_PALETTE = {
    'Angry': {'bgr': (78, 63, 244), 'hex': '#f43f5e', 'emoji': '😠'},
    'Fear': {'bgr': (246, 92, 139), 'hex': '#8b5cf6', 'emoji': '😨'},
    'Happy': {'bgr': (129, 185, 16), 'hex': '#10b981', 'emoji': '😃'},
    'Sad': {'bgr': (212, 182, 6), 'hex': '#06b6d4', 'emoji': '😢'},
    'Suprise': {'bgr': (11, 158, 245), 'hex': '#f59e0b', 'emoji': '😲'},
    'Surprise': {'bgr': (11, 158, 245), 'hex': '#f59e0b', 'emoji': '😲'},
    'Neutral': {'bgr': (184, 163, 148), 'hex': '#94a3b8', 'emoji': '😐'},
    'Disgust': {'bgr': (80, 160, 90), 'hex': '#5aa050', 'emoji': '🤢'},
    'Contempt': {'bgr': (150, 120, 200), 'hex': '#c87896', 'emoji': '😒'},
}


def palette_for(name):
    """Palette entry for a class name, case-insensitive ('happy' -> 'Happy')."""
    return EMOTION_PALETTE.get(name) or EMOTION_PALETTE.get(str(name).capitalize()) or {}

YUNET_MODEL_PATH = settings.MODEL_DIR / 'face_detection_yunet_2023mar.onnx'
PRIMARY_ONNX_MODEL_PATH = settings.MODEL_DIR / PRIMARY_MODEL_FILE


class FaceEngine:
    """Manages face detection and 5-class emotion classification pipelines."""

    def __init__(self):
        self.yunet_detector = None
        self.onnx_sessions = {}
        self._init_face_detector()
        self._init_emotion_models()

    def _init_face_detector(self):
        """Initialize the YuNet ONNX face detector if available."""
        if os.path.exists(YUNET_MODEL_PATH):
            try:
                self.yunet_detector = cv2.FaceDetectorYN_create(
                    model=str(YUNET_MODEL_PATH),
                    config='',
                    input_size=(320, 320),
                    score_threshold=0.6,
                    nms_threshold=0.3,
                    top_k=5000
                )
            except Exception as e:
                print(f"[FaceEngine] YuNet detector init notice: {e}")
                self.yunet_detector = None

    def _init_emotion_models(self):
        """Initialize ONNX inference sessions for available models."""
        model_dir = settings.MODEL_DIR
        for item in os.listdir(model_dir):
            if item.endswith('.onnx') and not item.startswith('face_detection'):
                path = os.path.join(model_dir, item)
                try:
                    opts = ort.SessionOptions()
                    opts.intra_op_num_threads = 2
                    available = ort.get_available_providers()
                    providers = [p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider') if p in available]
                    sess = ort.InferenceSession(path, sess_options=opts, providers=providers)
                    inputs = sess.get_inputs()
                    outputs = sess.get_outputs()
                    is_v2 = V2_META is not None and item == V2_META.get('onnx_file')
                    self.onnx_sessions[item] = {
                        'session': sess,
                        'input_name': inputs[0].name,
                        'input_shape': inputs[0].shape,
                        'output_name': outputs[0].name,
                        'output_shape': outputs[0].shape,
                        'kind': 'v2' if is_v2 else 'legacy',
                        'classes': list(V2_META['classes']) if is_v2 else LEGACY_CLASS_NAMES,
                        'image_size': int(V2_META['image_size']) if is_v2 else 48,
                        'outputs_probabilities': bool(is_v2 and V2_META.get('output') == 'probabilities'),
                        'face_margin': float(V2_META.get('face_margin', 0.2)) if is_v2 else 0.15,
                    }
                    print(f"[FaceEngine] Initialized ONNX model {item} -> input: {inputs[0].shape}, output: {outputs[0].shape}")
                except Exception as e:
                    print(f"[FaceEngine] Error loading {item}: {e}")

    def get_available_models(self):
        """List all trained model artifacts with metadata."""
        models = []
        model_dir = settings.MODEL_DIR
        for item in os.listdir(model_dir):
            if item.endswith(('.onnx', '.keras', '.tflite')) and not item.startswith('face_detection'):
                path = os.path.join(model_dir, item)
                size_mb = round(os.path.getsize(path) / (1024 * 1024), 2)
                is_primary = item == PRIMARY_MODEL_FILE
                meta = self.onnx_sessions.get(item, {})

                if meta.get('kind') == 'v2':
                    members = V2_META.get('members', [])
                    backbones = sorted({m.get('backbone', '?') for m in members})
                    acc = V2_META.get('metrics', {}).get('test_acc')
                    model_type = (f"Ensemble of {len(members)} ({', '.join(backbones)})"
                                  + (f", test acc {acc * 100:.1f}%" if acc else ""))
                    size = int(V2_META['image_size'])
                    input_shape = f"{size} x {size} (face crop)"
                    classes = list(V2_META['classes'])
                else:
                    model_type = f"ONNX {len(LEGACY_CLASS_NAMES)}-Class Model (48x48 RGB)" if "face_emotion" in item else \
                                 "ONNX Neural Network" if "onnx" in item else \
                                 "Keras Model" if "keras" in item else \
                                 "TFLite Edge Model"
                    input_shape = "48 x 48 (RGB)"
                    classes = LEGACY_CLASS_NAMES

                models.append({
                    'filename': item,
                    'name': item.replace('_', ' ').replace('.onnx', '').replace('.keras', '').replace('.tflite', '').title(),
                    'type': model_type,
                    'size_mb': size_mb,
                    'classes': classes,
                    'input_shape': input_shape,
                    'is_default': is_primary
                })
        
        models.sort(key=lambda m: not m['is_default'])
        return models

    def decode_image(self, image_data):
        """Decodes base64 string, file object, or bytes into an OpenCV BGR numpy array."""
        if isinstance(image_data, str):
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            img_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif hasattr(image_data, 'read'):
            img_bytes = image_data.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_data, (bytes, bytearray)):
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_data, np.ndarray):
            img = image_data
        else:
            raise ValueError("Unsupported image data type")

        if img is None:
            raise ValueError("Failed to decode image data into valid image")
        return img

    def detect_faces(self, bgr_image):
        """
        Detect faces in the image using YuNet or fallback center crop.
        Returns a list of dicts with bbox [x, y, w, h], score, and landmarks.
        """
        h, w = bgr_image.shape[:2]
        faces = []

        if self.yunet_detector is not None:
            try:
                self.yunet_detector.setInputSize((w, h))
                _, detected = self.yunet_detector.detect(bgr_image)
                if detected is not None:
                    for det in detected:
                        box = det[0:4].astype(int)
                        score = float(det[-1])
                        bx = max(0, int(box[0]))
                        by = max(0, int(box[1]))
                        bw = min(w - bx, int(box[2]))
                        bh = min(h - by, int(box[3]))
                        if bw > 15 and bh > 15:
                            faces.append({
                                'box': [bx, by, bw, bh],
                                'confidence': score,
                                'landmarks': det[4:14].reshape((5, 2)).tolist() if len(det) >= 14 else []
                            })
            except Exception as e:
                print(f"[FaceEngine] YuNet detection error: {e}")

        # Fallback to whole frame / center region
        if not faces:
            min_dim = min(h, w)
            cw, ch = int(min_dim * 0.8), int(min_dim * 0.8)
            cx, cy = (w - cw) // 2, (h - ch) // 2
            faces.append({
                'box': [cx, cy, cw, ch],
                'confidence': 0.85,
                'landmarks': [],
                'is_fallback': True
            })

        return faces

    def _resolve_model(self, model_name):
        """Session key to use: the requested model if loaded, else the default model."""
        if model_name in self.onnx_sessions:
            return model_name
        if PRIMARY_MODEL_FILE in self.onnx_sessions:
            return PRIMARY_MODEL_FILE
        return next(iter(self.onnx_sessions), None)

    def preprocess_face(self, bgr_image, box, padding=0.15, model_key=None):
        """
        Crops the face and builds the input tensor for the given model.
        v2 models: square crop with the training margin, size from deployment_metadata.json.
        legacy models: padded box resized to 48x48 RGB NCHW.
        """
        meta = self.onnx_sessions.get(model_key or self._resolve_model(None), {})
        if meta.get('kind') == 'v2':
            h_img, w_img = bgr_image.shape[:2]
            small = min(h_img, w_img) < int(V2_META.get('assume_crop_below', 0))
            if small or list(box) == [0, 0, w_img, h_img]:
                face_crop = center_square(bgr_image)       # the image is a face crop already
            else:
                face_crop, _ = square_crop(bgr_image, box, meta['face_margin'])
            if face_crop.size == 0:
                face_crop = center_square(bgr_image)
            return face_crop, preprocess_faces([face_crop], meta['image_size'])

        h_img, w_img = bgr_image.shape[:2]
        x, y, w, h = box

        pad_x = int(w * padding)
        pad_y = int(h * padding)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w_img, x + w + pad_x)
        y2 = min(h_img, y + h + pad_y)

        face_crop = bgr_image[y1:y2, x1:x2]
        if face_crop.size == 0:
            face_crop = bgr_image

        # 48x48 RGB in NCHW format
        face_resized = cv2.resize(face_crop, (48, 48), interpolation=cv2.INTER_AREA)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Normalized float32 tensor [0.0, 1.0]
        tensor = face_rgb.astype(np.float32) / 255.0
        # (48, 48, 3) -> (3, 48, 48) -> (1, 3, 48, 48)
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)

        return face_crop, tensor

    def predict_expression(self, face_tensor, model_name=None):
        """
        Predicts emotion probabilities with the requested model (default: v2 model if installed,
        else face_emotion_model.onnx). v2 models output probabilities; legacy models output logits.
        """
        target_key = self._resolve_model(model_name)
        meta = self.onnx_sessions.get(target_key, {})
        classes = meta.get('classes', CLASS_NAMES)

        if meta:
            try:
                sess = meta['session']
                raw_out = sess.run(None, {meta['input_name']: face_tensor})[0][0]

                if meta.get('outputs_probabilities'):
                    probabilities = np.clip(raw_out[:len(classes)].astype(np.float64), 0.0, 1.0)
                    probabilities = probabilities / max(probabilities.sum(), 1e-12)
                else:
                    logits = raw_out[:len(classes)].astype(np.float64)
                    exp_logits = np.exp(logits - np.max(logits))
                    probabilities = exp_logits / np.sum(exp_logits)
            except Exception as e:
                print(f"[FaceEngine] Inference error on {target_key}: {e}")
                probabilities = np.ones(len(classes), dtype=np.float32) / len(classes)
        else:
            probabilities = np.ones(len(classes), dtype=np.float32) / len(classes)

        prob_dict = {
            classes[i]: float(probabilities[i])
            for i in range(len(classes))
        }

        top_idx = int(np.argmax(probabilities))
        primary_emotion = classes[top_idx]
        confidence = float(probabilities[top_idx])

        return primary_emotion, confidence, prob_dict

    def process_frame(self, image_data, model_name=None, draw_annotations=True):
        """
        Complete end-to-end processing pipeline for an incoming frame/image.
        Returns detection summary, faces, 5-class probabilities, and base64 annotated image.
        """
        start_time = time.time()
        bgr_image = self.decode_image(image_data)
        model_key = self._resolve_model(model_name)
        model_name = model_key or model_name
        h_img, w_img = bgr_image.shape[:2]
        if (self.onnx_sessions.get(model_key, {}).get('kind') == 'v2'
                and min(h_img, w_img) < int(V2_META.get('assume_crop_below', 0))):
            # tiny image (e.g. 48x48): it is a face crop already, like the training images
            faces_detected = [{'box': [0, 0, w_img, h_img], 'confidence': 1.0, 'landmarks': [], 'is_fallback': True}]
        else:
            faces_detected = self.detect_faces(bgr_image)

        results = []
        annotated_bgr = bgr_image.copy() if draw_annotations else None

        for idx, face_info in enumerate(faces_detected):
            box = face_info['box']
            face_crop, face_tensor = self.preprocess_face(bgr_image, box, model_key=model_key)

            primary_emotion, confidence, probabilities = self.predict_expression(
                face_tensor, model_name=model_key
            )

            # Encode face crop thumbnail
            _, crop_buf = cv2.imencode('.jpg', face_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            crop_b64 = "data:image/jpeg;base64," + base64.b64encode(crop_buf).decode('utf-8')

            emoji = palette_for(primary_emotion).get('emoji', '😐')
            color = palette_for(primary_emotion).get('hex', '#38bdf8')

            face_result = {
                'face_id': idx + 1,
                'box': box,
                'primary_emotion': primary_emotion,
                'confidence': round(confidence * 100, 1),
                'probabilities': {k: round(v * 100, 1) for k, v in probabilities.items()},
                'emoji': emoji,
                'color': color,
                'face_crop_base64': crop_b64
            }
            results.append(face_result)

            if draw_annotations:
                x, y, w, h = box
                bgr_color = palette_for(primary_emotion).get('bgr', (255, 100, 0))
                
                # Draw rounded corner bounding box
                cv2.rectangle(annotated_bgr, (x, y), (x + w, y + h), bgr_color, 2)
                
                # Draw corner accents
                corner_len = min(20, w // 4, h // 4)
                cv2.line(annotated_bgr, (x, y), (x + corner_len, y), (255, 255, 255), 3)
                cv2.line(annotated_bgr, (x, y), (x, y + corner_len), (255, 255, 255), 3)
                cv2.line(annotated_bgr, (x + w, y), (x + w - corner_len, y), (255, 255, 255), 3)
                cv2.line(annotated_bgr, (x + w, y), (x + w, y + corner_len), (255, 255, 255), 3)
                cv2.line(annotated_bgr, (x, y + h), (x + corner_len, y + h), (255, 255, 255), 3)
                cv2.line(annotated_bgr, (x, y + h), (x, y + h - corner_len), (255, 255, 255), 3)
                cv2.line(annotated_bgr, (x + w, y + h), (x + w - corner_len, y + h), (255, 255, 255), 3)
                cv2.line(annotated_bgr, (x + w, y + h), (x + w, y + corner_len), (255, 255, 255), 3)

                # Draw label badge
                label = f"{primary_emotion} {round(confidence * 100)}%"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                badge_y = max(y - 10, lh + 10)
                cv2.rectangle(annotated_bgr, (x, badge_y - lh - 6), (x + lw + 12, badge_y + 4), bgr_color, -1)
                cv2.putText(annotated_bgr, label, (x + 6, badge_y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        inference_time_ms = (time.time() - start_time) * 1000

        annotated_b64 = None
        if draw_annotations and annotated_bgr is not None:
            _, img_buf = cv2.imencode('.jpg', annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(img_buf).decode('utf-8')

        primary = results[0] if results else {
            'primary_emotion': 'Neutral',
            'confidence': 0.0,
            'probabilities': {k: round(100.0 / len(CLASS_NAMES), 1) for k in CLASS_NAMES},
            'emoji': '😐',
            'color': '#94a3b8'
        }

        return {
            'success': True,
            'face_count': len(results),
            'primary_emotion': primary['primary_emotion'],
            'confidence': primary['confidence'],
            'probabilities': primary.get('probabilities', {}),
            'emoji': primary.get('emoji', '😐'),
            'color': primary.get('color', '#38bdf8'),
            'faces': results,
            'annotated_image': annotated_b64,
            'inference_time_ms': round(inference_time_ms, 1),
            'model_name': model_name,
        }


# Global singleton engine instance
engine = FaceEngine()
