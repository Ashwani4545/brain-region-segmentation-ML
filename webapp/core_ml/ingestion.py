import os
import cv2
import pydicom
from core_ml.inference_service import get_inference_service
from core_ml.chest_xray import get_chest_xray_analyzer
from core_ml.ecg import get_ecg_analyzer
from core_ml.blood_test import get_blood_test_analyzer

class IngestionService:
    def __init__(self):
        pass

    def route_file(self, file_path: str, output_dir: str) -> dict:
        """
        Detect file modality and route to the correct parser/model.
        
        Returns:
            dict containing:
                'modality': str ('CT', 'CXR', 'ECG', 'BLOOD_TEST')
                'result': dict (specific findings of the routed model)
        """
        ext = os.path.splitext(file_path.lower())[1]
        
        # 1. PDF Documents -> Blood/Lab test
        if ext == '.pdf':
            analyzer = get_blood_test_analyzer()
            res = analyzer.process_pdf(file_path)
            return {
                'modality': 'BLOOD_TEST',
                'result': res
            }
            
        # 2. DICOM Medical Files -> Check DICOM modality header tags
        if ext == '.dcm':
            try:
                ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                modality_tag = str(getattr(ds, 'Modality', '')).upper()
                
                if 'ECG' in modality_tag:
                    analyzer = get_ecg_analyzer()
                    res = analyzer.process_image(file_path, output_dir)
                    return {
                        'modality': 'ECG',
                        'result': res
                    }
                elif 'CR' in modality_tag or 'DX' in modality_tag:
                    analyzer = get_chest_xray_analyzer()
                    res = analyzer.process_image(file_path, output_dir)
                    return {
                        'modality': 'CXR',
                        'result': res
                    }
            except Exception as e:
                print(f"[Ingestion] DICOM header read failed, falling back: {e}")
                
            # Default DICOM to Brain CT
            service = get_inference_service()
            res = service.process_image(file_path, output_dir)
            return {
                'modality': 'CT',
                'result': res
            }

        # 3. Standard Images (JPG/PNG) -> Run keyword and visual heuristics router
        if ext in ['.jpg', '.jpeg', '.png']:
            filename = os.path.basename(file_path).lower()
            
            # Key 1: Filename Keyword checks
            if any(w in filename for w in ['ecg', 'electrocardiogram', 'rhythm', 'wave', 'lead']):
                analyzer = get_ecg_analyzer()
                res = analyzer.process_image(file_path, output_dir)
                return {
                    'modality': 'ECG',
                    'result': res
                }
            elif any(w in filename for w in ['xray', 'cxr', 'chest', 'lung', 'pneumonia', 'pulmonary']):
                analyzer = get_chest_xray_analyzer()
                res = analyzer.process_image(file_path, output_dir)
                return {
                    'modality': 'CXR',
                    'result': res
                }
            elif any(w in filename for w in ['brain', 'ct', 'ncct', 'head', 'mri', 'stroke', 'slice']):
                service = get_inference_service()
                res = service.process_image(file_path, output_dir)
                return {
                    'modality': 'CT',
                    'result': res
                }
                
            # Key 2: Visual Aspect-Ratio Heuristics checks
            try:
                img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    h, w = img.shape[:2]
                    aspect_ratio = w / h
                    
                    # ECG reports are typically wide scans
                    if aspect_ratio >= 1.6:
                        analyzer = get_ecg_analyzer()
                        res = analyzer.process_image(file_path, output_dir)
                        return {
                            'modality': 'ECG',
                            'result': res
                        }
                    # Chest X-rays are typically vertical rectangular lung captures
                    elif aspect_ratio <= 0.88:
                        analyzer = get_chest_xray_analyzer()
                        res = analyzer.process_image(file_path, output_dir)
                        return {
                            'modality': 'CXR',
                            'result': res
                        }
            except Exception as e:
                print(f"[Ingestion] Image dimension check failed: {e}")
                
            # Default to standard Brain CT pipeline
            service = get_inference_service()
            res = service.process_image(file_path, output_dir)
            return {
                'modality': 'CT',
                'result': res
            }
            
        raise ValueError(f"Unsupported report format extension: {ext}")

_service = None

def get_ingestion_service() -> IngestionService:
    global _service
    if _service is None:
        _service = IngestionService()
    return _service
