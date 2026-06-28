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


class PatientGuidance(models.Model):
    scan = models.OneToOneField(PatientScan, on_delete=models.CASCADE, related_name='guidance')
    diet_plan = models.JSONField()            # foods to eat, avoid, and portion advice
    exercise_plan = models.JSONField()        # safe physical activities
    lifestyle_plan = models.JSONField()       # stress, sleep, and warning red flags
    doctor_questions = models.JSONField()     # 5-8 consult questions
    specialist_routing = models.JSONField()   # specialty and referral tips
    created_at = models.DateTimeField(auto_now_add=True)


class PatientRiskProfile(models.Model):
    scan = models.OneToOneField(PatientScan, on_delete=models.CASCADE, related_name='risk_profile')
    cv_risk_score = models.FloatField()         # Cardiovascular risk percentage
    cv_risk_grade = models.CharField(max_length=20)
    diabetes_risk_score = models.FloatField()   # Diabetes risk percentage
    diabetes_risk_grade = models.CharField(max_length=20)
    stroke_risk_score = models.FloatField()     # Stroke risk percentage
    stroke_risk_grade = models.CharField(max_length=20)
    detailed_metrics = models.JSONField()       # age, BP, BMI, smoking status used
    created_at = models.DateTimeField(auto_now_add=True)


class TelehealthConsultation(models.Model):
    scan = models.OneToOneField(PatientScan, on_delete=models.CASCADE, related_name='consultation')
    assigned_doctor = models.CharField(max_length=100) # Principal doctor
    specialist_panel = models.JSONField(default=list)  # Invited doctors list (e.g. ["Dr. Priya Patel (Cardiology)"])
    status = models.CharField(max_length=20, default='REQUESTED') # REQUESTED, ACTIVE, COMPLETED
    clinical_notes = models.TextField(blank=True, null=True)     # Doctor sign-off commentary
    signed_off_by = models.CharField(max_length=100, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)


class ConsultationMessage(models.Model):
    consultation = models.ForeignKey(TelehealthConsultation, on_delete=models.CASCADE, related_name='messages')
    sender_name = models.CharField(max_length=100) # Patient name or Doctor name
    is_doctor = models.BooleanField(default=False)
    message = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
