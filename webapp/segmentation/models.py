import random
from datetime import datetime
from django.db import models
from django.contrib.auth.models import User

class PatientScan(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    patient_id = models.CharField(max_length=50, unique=True, editable=False)
    scan_name = models.CharField(max_length=255)
    modality = models.CharField(max_length=20, default='NCCT')
    original_image = models.CharField(max_length=500)
    mask_image = models.CharField(max_length=500)
    overlay_image = models.CharField(max_length=500)
    confidence = models.FloatField()  # raw percentage, e.g., 36.2
    detected = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    notes = models.TextField(blank=True, null=True)

    def save(self, *args, **kwargs):
        if not self.patient_id:
            year = datetime.now().year
            num = random.randint(10000, 99999)
            self.patient_id = f"ND-{year}-{num}"
        super().save(*args, **kwargs)

    class Meta:
        ordering = ['-created_at']


class ChatMessage(models.Model):
    scan = models.ForeignKey(PatientScan, on_delete=models.CASCADE, related_name='chat_messages')
    role = models.CharField(max_length=20)  # 'user' or 'assistant'
    message = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    emotion_detected = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        ordering = ['timestamp']
