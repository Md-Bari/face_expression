"""
Face Detection and Emotion Recognition Engine.
Integrates OpenCV YuNet Face Detector with the ONNX Facial Emotion Recognition Model (face_emotion_model.onnx).
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

# Emotion Class Names
CLASSES_PATH = settings.MODEL_DIR / 'class_names.json'
try:
    with open(CLASSES_PATH, 'r') as f:
        CLASS_NAMES = json.load(f)
except Exception:
    CLASS_NAMES = ["Angry", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Color palette for emotions (BGR for OpenCV, Hex for UI)
EMOTION_PALETTE = {
    'Angry': {'bgr': (78, 63, 244), 'hex': '#f43f5e', 'emoji': '😠'},
    'Contempt': {'bgr': (180, 72, 236), 'hex': '#ec4899', 'emoji': '😒'},
    'Disgust': {'bgr': (22, 204, 132), 'hex': '#84cc16', 'emoji': '🤢'},
    'Fear': {'bgr': (246, 92, 139), 'hex': '#8b5cf6', 'emoji': '😨'},
    'Happy': {'bgr': (129, 185, 16), 'hex': '#10b981', 'emoji': '😃'},
    'Neutral': {'bgr': (184, 163, 148), 'hex': '#94a3b8', 'emoji': '😐'},
    'Sad': {'bgr': (212, 182, 6), 'hex': '#06b6d4', 'emoji': '😢'},
    'Surprise': {'bgr': (11, 158, 245), 'hex': '#f59e0b', 'emoji': '😲'},
    'Suprise': {'bgr': (11, 158, 245), 'hex': '#f59e0b', 'emoji': '😲'},
}

YUNET_MODEL_PATH = settings.MODEL_DIR / 'face_detection_yunet_2023mar.onnx'
PRIMARY_ONNX_MODEL_PATH = settings.MODEL_DIR / 'face_emotion_model.onnx'


class FaceEngine:
    """Manages face detection and emotion classification pipelines."""

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
                    sess = ort.InferenceSession(path, sess_options=opts)
                    inputs = sess.get_inputs()
                    outputs = sess.get_outputs()
                    self.onnx_sessions[item] = {
                        'session': sess,
                        'input_name': inputs[0].name,
                        'input_shape': inputs[0].shape,
                        'output_name': outputs[0].name,
                        'output_shape': outputs[0].shape,
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
                
                is_primary = item == 'face_emotion_model.onnx'
                
                input_desc = "48 x 48 (RGB)" if "face_emotion" in item else "150 x 150 (Gray)"
                
                model_type = "ONNX Deep CNN (48x48 RGB)" if "face_emotion" in item else \
                             "ONNX Neural Network" if "onnx" in item else \
                             "Keras Sequential CNN" if "keras" in item else \
                             "TFLite Edge Model"
                
                models.append({
                    'filename': item,
                    'name': item.replace('_', ' ').replace('.onnx', '').replace('.keras', '').replace('.tflite', '').title(),
                    'type': model_type,
                    'size_mb': size_mb,
                    'classes': CLASS_NAMES,
                    'input_shape': input_desc,
                    'is_default': is_primary
                })
        
        # Sort so default primary model is first
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

        # If no face detected, fallback to center region
        if not faces:
            min_dim = min(h, w)
            cw, ch = int(min_dim * 0.75), int(min_dim * 0.75)
            cx, cy = (w - cw) // 2, (h - ch) // 2
            faces.append({
                'box': [cx, cy, cw, ch],
                'confidence': 0.85,
                'landmarks': [],
                'is_fallback': True
            })

        return faces

    def preprocess_face(self, bgr_image, box, target_model='face_emotion_model.onnx', padding=0.15):
        """
        Crops face with padding and converts to format expected by target model.
        For face_emotion_model.onnx: (1, 3, 48, 48) RGB float32.
        For 150x150 model: (1, 150, 150, 1) Grayscale float32.
        """
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

        if 'face_emotion_model' in target_model or 'onnx' in target_model and 'face_emotion' in target_model:
            # 48x48 RGB in NCHW format
            face_resized = cv2.resize(face_crop, (48, 48), interpolation=cv2.INTER_AREA)
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
            # Standard PyTorch float tensor normalized to [0, 1]
            tensor = face_rgb.astype(np.float32) / 255.0
            # NCHW: (48, 48, 3) -> (3, 48, 48) -> (1, 3, 48, 48)
            tensor = np.transpose(tensor, (2, 0, 1))
            tensor = np.expand_dims(tensor, axis=0)
        else:
            # 150x150 Grayscale
            face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(face_gray, (150, 150), interpolation=cv2.INTER_AREA)
            tensor = (face_resized.astype(np.float32) / 255.0).reshape(1, 150, 150, 1)

        return face_crop, tensor

    def predict_expression(self, face_tensor, model_name='face_emotion_model.onnx'):
        """
        Predicts emotion probabilities using the specified ONNX model.
        """
        # Resolve model name
        target_key = 'face_emotion_model.onnx' if 'face_emotion' in model_name or model_name not in self.onnx_sessions else model_name
        
        if target_key not in self.onnx_sessions and self.onnx_sessions:
            target_key = list(self.onnx_sessions.keys())[0]

        if target_key in self.onnx_sessions:
            try:
                model_meta = self.onnx_sessions[target_key]
                sess = model_meta['session']
                input_name = model_meta['input_name']
                
                # Verify input shape
                expected_shape = model_meta['input_shape']
                if len(expected_shape) == 4 and expected_shape[1] == 3 and face_tensor.shape[1] != 3:
                    # In case shape is (1, 150, 150, 1), reshape to (1, 3, 48, 48)
                    face_tensor = np.repeat(face_tensor, 3, axis=1) if face_tensor.shape[1] == 1 else face_tensor

                outputs = sess.run(None, {input_name: face_tensor})
                raw_out = outputs[0][0]  # Array of logits or probs
                
                # Check output dimension
                num_classes = len(CLASS_NAMES)
                if len(raw_out) >= num_classes:
                    logits = raw_out[:num_classes].astype(np.float64)
                else:
                    logits = np.zeros(num_classes, dtype=np.float64)
                    logits[:len(raw_out)] = raw_out

                # Apply Softmax with numerical stability
                exp_logits = np.exp(logits - np.max(logits))
                probabilities = exp_logits / np.sum(exp_logits)
            except Exception as e:
                print(f"[FaceEngine] Inference error on {target_key}: {e}")
                probabilities = np.ones(len(CLASS_NAMES), dtype=np.float32) / len(CLASS_NAMES)
        else:
            probabilities = np.ones(len(CLASS_NAMES), dtype=np.float32) / len(CLASS_NAMES)

        prob_dict = {
            CLASS_NAMES[i]: float(probabilities[i])
            for i in range(len(CLASS_NAMES))
        }

        top_idx = int(np.argmax(probabilities))
        primary_emotion = CLASS_NAMES[top_idx]
        confidence = float(probabilities[top_idx])

        return primary_emotion, confidence, prob_dict

    def process_frame(self, image_data, model_name='face_emotion_model.onnx', draw_annotations=True):
        """
        Complete end-to-end processing pipeline for an incoming frame/image.
        Returns detection summary, faces, probabilities, and base64 annotated image.
        """
        start_time = time.time()
        bgr_image = self.decode_image(image_data)
        faces_detected = self.detect_faces(bgr_image)

        results = []
        annotated_bgr = bgr_image.copy() if draw_annotations else None

        for idx, face_info in enumerate(faces_detected):
            box = face_info['box']
            face_crop, face_tensor = self.preprocess_face(bgr_image, box, target_model=model_name)
            
            primary_emotion, confidence, probabilities = self.predict_expression(
                face_tensor, model_name=model_name
            )

            # Encode face crop thumbnail
            _, crop_buf = cv2.imencode('.jpg', face_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            crop_b64 = "data:image/jpeg;base64," + base64.b64encode(crop_buf).decode('utf-8')

            emoji = EMOTION_PALETTE.get(primary_emotion, {}).get('emoji', '😐')
            color = EMOTION_PALETTE.get(primary_emotion, {}).get('hex', '#38bdf8')

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
                bgr_color = EMOTION_PALETTE.get(primary_emotion, {}).get('bgr', (255, 100, 0))
                
                # Draw rounded corner bounding box
                cv2.rectangle(annotated_bgr, (x, y), (x + w, y + h), bgr_color, 2)
                
                # Draw corner accents
                corner_len = min(20, w // 4, h // 4)
                cv2.line(annotated_bgr, (x, y), (x + corner_len, y), (255, 255, 255), 3)
                cv2.line(annotated_bgr, (x, y), (x, y + corner_len), (255, 255, 255), 3)
                cv2.line(annotated_bgr, (x + w, y), (x + w - corner_len, y), (255, 255, 255), 3)
                cv2.line(annotated_bgr, (x + w, y), (x + w, y + corner_len), (255, 255, 255), 3)

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
            'probabilities': {k: 12.5 for k in CLASS_NAMES},
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
