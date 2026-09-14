from django.db import models
from django.utils import timezone
import json


class EmotionDetectionRecord(models.Model):
    """Stores individual facial expression recognition predictions and analytics."""
    SOURCE_CHOICES = [
        ('webcam_live', 'Live Webcam Stream'),
        ('webcam_snap', 'Webcam Snapshot'),
        ('image_upload', 'Image File Upload'),
    ]

    EMOTION_CHOICES = [
        ('Angry', 'Angry'),
        ('Fear', 'Fear'),
        ('Happy', 'Happy'),
        ('Sad', 'Sad'),
        ('Suprise', 'Suprise'),
    ]

    created_at = models.DateTimeField(default=timezone.now, db_index=True)
    source = models.CharField(max_length=20, choices=SOURCE_CHOICES, default='image_upload')
    model_name = models.CharField(max_length=64, default='best_finetuned.keras')
    
    # Primary detected emotion
    primary_emotion = models.CharField(max_length=20, choices=EMOTION_CHOICES, db_index=True)
    confidence = models.FloatField(help_text='Confidence score between 0.0 and 1.0')
    
    # Full probability distribution across all 5 classes stored as JSON
    probabilities = models.JSONField(default=dict, help_text='Dict of emotion to confidence')
    
    # Face bounding box [x, y, width, height]
    bounding_box = models.JSONField(default=list, blank=True, help_text='[x, y, w, h]')
    
    # Images (Optional: stored if user saves snapshot or upload)
    image_file = models.ImageField(upload_to='detections/%Y/%m/%d/', blank=True, null=True)
    face_crop = models.ImageField(upload_to='faces/%Y/%m/%d/', blank=True, null=True)
    
    # Performance metrics
    inference_time_ms = models.FloatField(default=0.0)
    face_count = models.IntegerField(default=1)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Emotion Detection Record'
        verbose_name_plural = 'Emotion Detection Records'

    def __str__(self):
        return f"{self.primary_emotion} ({self.confidence:.1%}) - {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}"

    def to_dict(self):
        return {
            'id': self.id,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'timestamp': self.created_at.isoformat(),
            'source': self.source,
            'model_name': self.model_name,
            'primary_emotion': self.primary_emotion,
            'confidence': round(self.confidence * 100, 1),
            'probabilities': {k: round(v * 100, 1) for k, v in self.probabilities.items()} if self.probabilities else {},
            'bounding_box': self.bounding_box,
            'image_url': self.image_file.url if self.image_file else None,
            'face_crop_url': self.face_crop.url if self.face_crop else None,
            'inference_time_ms': round(self.inference_time_ms, 1),
            'face_count': self.face_count,
        }
