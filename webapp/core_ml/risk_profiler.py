import os

class ClinicalRiskProfiler:
    def __init__(self):
        pass

    def calculate_profile(self, scan, overrides=None) -> dict:
        """
        Calculate Cardiovascular, Diabetes, and Stroke risk scores.
        Extracts values from the scan (and blood tests if available)
        and integrates user-provided overrides.
        """
        overrides = overrides or {}
        
        # 1. Gather inputs (with fallback defaults)
        age = float(overrides.get('age', 52))
        sbp = float(overrides.get('systolic_bp', 128))
        bmi = float(overrides.get('bmi', 24.2))
        smoking = str(overrides.get('smoking', 'No')).strip().lower() == 'yes'
        
        # Pull markers from blood test if present
        glucose = 90.0
        cholesterol = 180.0
        
        # Check if the scan has metrics logged (e.g. from Blood Test)
        # Note: scan.notes contains findings text, we can check or query
        if scan.modality == 'BLOOD_TEST':
            # Try to extract from scan metrics if stored or parse from notes
            notes = (scan.notes or '').lower()
            # Search for numbers in notes if metrics are not fully parsed
            if 'glucose' in notes:
                # e.g., "Glucose detected (142.0 mg/dL)"
                import re
                gluc_match = re.search(r'glucose.*?(\d+\.?\d*)', notes)
                if gluc_match:
                    glucose = float(gluc_match.group(1))
            if 'cholesterol' in notes:
                import re
                chol_match = re.search(r'cholesterol.*?(\d+\.?\d*)', notes)
                if chol_match:
                    cholesterol = float(chol_match.group(1))
                    
        # 2. Cardiovascular Risk (ASCVD 10-Year Risk Approximation)
        cv_score = self._compute_cv_risk(age, sbp, cholesterol, glucose, smoking)
        cv_grade = "Low"
        if cv_score >= 20.0:
            cv_grade = "High"
        elif cv_score >= 7.5:
            cv_grade = "Intermediate"
        elif cv_score >= 5.0:
            cv_grade = "Borderline"
            
        # 3. Diabetes Risk (FINDRISC/ADA Risk Approximation)
        db_score = self._compute_diabetes_risk(age, bmi, glucose)
        db_grade = "Low"
        if db_score >= 30.0:
            db_grade = "High"
        elif db_score >= 10.0:
            db_grade = "Moderate"
            
        # 4. Stroke Risk Score (Clinical multi-factor model)
        stroke_score = self._compute_stroke_risk(scan, age, sbp, smoking)
        stroke_grade = "Low"
        if stroke_score >= 15.0:
            stroke_grade = "High"
        elif stroke_score >= 5.0:
            stroke_grade = "Moderate"
            
        return {
            'cv_risk_score': round(cv_score, 2),
            'cv_risk_grade': cv_grade,
            'diabetes_risk_score': round(db_score, 2),
            'diabetes_risk_grade': db_grade,
            'stroke_risk_score': round(stroke_score, 2),
            'stroke_risk_grade': stroke_grade,
            'detailed_metrics': {
                'age': age,
                'systolic_bp': sbp,
                'bmi': bmi,
                'smoking': 'Yes' if smoking else 'No',
                'glucose': glucose,
                'cholesterol': cholesterol
            }
        }

    def _compute_cv_risk(self, age, sbp, cholesterol, glucose, smoking) -> float:
        """
        ASCVD Risk Estimation.
        Baseline: 1.5% for age 50. Increases exponentially.
        """
        # Age component
        base_risk = 1.0 * (1.085 ** (age - 45))
        
        # BP component (+15% per 10mmHg above 120)
        bp_factor = 1.0
        if sbp > 120:
            bp_factor += ((sbp - 120) / 10) * 0.15
            
        # Cholesterol component (+15% per 40mg/dL above 180)
        chol_factor = 1.0
        if cholesterol > 180:
            chol_factor += ((cholesterol - 180) / 40) * 0.15
            
        # Glucose factor (+30% if glucose indicates pre-diabetes/diabetes)
        gluc_factor = 1.0
        if glucose > 126:
            gluc_factor = 1.6
        elif glucose > 100:
            gluc_factor = 1.25
            
        # Smoking factor (increases risk by 1.8x)
        smoke_factor = 1.8 if smoking else 1.0
        
        risk = base_risk * bp_factor * chol_factor * gluc_factor * smoke_factor
        return min(risk, 95.0)

    def _compute_diabetes_risk(self, age, bmi, glucose) -> float:
        """
        FINDRISC / ADA risk model.
        Fasting glucose is the primary driver. Obesity (BMI) and age act as accelerators.
        """
        # Primary driver: fasting glucose
        if glucose >= 126:
            base_risk = 75.0
        elif glucose >= 100:
            base_risk = 35.0
        else:
            base_risk = 4.0
            
        # Add BMI modifiers
        bmi_modifier = 0.0
        if bmi >= 30: # Obese
            bmi_modifier = 20.0
        elif bmi >= 25: # Overweight
            bmi_modifier = 8.0
            
        # Add Age modifiers
        age_modifier = 0.0
        if age >= 64:
            age_modifier = 12.0
        elif age >= 45:
            age_modifier = 5.0
            
        return min(base_risk + bmi_modifier + age_modifier, 98.0)

    def _compute_stroke_risk(self, scan, age, sbp, smoking) -> float:
        """
        Clinical stroke risk.
        Primary drivers: Brain CT hypodensities, Blood pressure, Age, and Cardiac/Pulmonary conditions.
        """
        # Base age factor
        risk = 1.2 * (1.065 ** (age - 45))
        
        # Systolic BP impact (essential stroke trigger)
        if sbp >= 160:
            risk += 35.0
        elif sbp >= 140:
            risk += 15.0
        elif sbp >= 130:
            risk += 5.0
            
        # Imaging Finding: Brain CT lesion/hypodensity
        if scan.modality == 'CT' and scan.detected:
            # Add significant risk for active hypodensity + proportional size
            risk += 30.0 + (scan.confidence * 1.2)
            
        # Cardiovascular/Pulmonary link:
        # 1. Cardiomegaly/ECG arrhythmia increases stroke thromboembolic risk
        if scan.modality == 'ECG' and scan.detected:
            risk += 15.0 # Arrhythmia thromboembolism
        elif scan.modality == 'CXR' and scan.detected:
            # Check notes for cardiomegaly
            notes = (scan.notes or '').lower()
            if 'cardiomegaly' in notes or 'enlarged' in notes:
                risk += 12.0
                
        # Smoking factor
        if smoking:
            risk *= 1.5
            
        return min(risk, 99.0)

_profiler = None

def get_risk_profiler() -> ClinicalRiskProfiler:
    global _profiler
    if _profiler is None:
        _profiler = ClinicalRiskProfiler()
    return _profiler
