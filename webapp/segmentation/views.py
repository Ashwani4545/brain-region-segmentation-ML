import os
import uuid
import random
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from django.db.models import Avg
from .models import PatientScan

# Allowed file extensions for upload validation
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.dcm'}
MAX_UPLOAD_MB = 20

def landing_page(request):
    return render(request, 'segmentation/index.html')

def app_page(request):
    return render(request, 'segmentation/app.html')

def dashboard_page(request):
    scans = PatientScan.objects.all()
    total_scans = scans.count()
    abnormal_scans = scans.filter(detected=True).count()
    normal_scans = total_scans - abnormal_scans
    
    avg_confidence = scans.aggregate(Avg('confidence'))['confidence__avg'] or 0.0
    abnormal_rate = (abnormal_scans / total_scans * 100) if total_scans > 0 else 0.0
    
    # Get last 5 scans
    recent_scans = scans[:5]
    
    context = {
        'total_scans': total_scans,
        'abnormal_scans': abnormal_scans,
        'normal_scans': normal_scans,
        'avg_confidence': f"{avg_confidence:.2f}%",
        'abnormal_rate': f"{abnormal_rate:.1f}%",
        'recent_scans': recent_scans,
    }
    return render(request, 'segmentation/dashboard.html', context)

def registry_page(request):
    query = request.GET.get('q', '')
    scans = PatientScan.objects.all()
    if query:
        scans = scans.filter(patient_id__icontains=query) | scans.filter(scan_name__icontains=query)
    
    return render(request, 'segmentation/registry.html', {'scans': scans, 'query': query})

def settings_page(request):
    if request.method == 'POST':
        try:
            request.session['hu_lo'] = float(request.POST.get('hu_lo', 15.0))
            request.session['hu_hi'] = float(request.POST.get('hu_hi', 35.0))
            request.session['min_area_dicom'] = int(request.POST.get('min_area_dicom', 50))
            request.session['min_area_image'] = int(request.POST.get('min_area_image', 30))
            request.session['save_success'] = True
        except ValueError:
            request.session['save_error'] = "Invalid input values. Please check numeric formatting."
        return redirect('settings')
        
    context = {
        'hu_lo': request.session.get('hu_lo', 15.0),
        'hu_hi': request.session.get('hu_hi', 35.0),
        'min_area_dicom': request.session.get('min_area_dicom', 50),
        'min_area_image': request.session.get('min_area_image', 30),
        'save_success': request.session.pop('save_success', False),
        'save_error': request.session.pop('save_error', None),
    }
    return render(request, 'segmentation/settings.html', context)

@csrf_exempt
def delete_scan_api(request, scan_id):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Method not allowed.'})
    try:
        scan = PatientScan.objects.get(id=scan_id)
        # Delete associated media files from storage
        for img_url in [scan.original_image, scan.mask_image, scan.overlay_image]:
            if img_url.startswith('/media/'):
                relative_path = img_url.replace('/media/', '')
                file_path = os.path.join(settings.MEDIA_ROOT, relative_path)
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception:
                        pass
        scan.delete()
        return JsonResponse({'success': True})
    except PatientScan.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Scan record not found.'})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

@csrf_exempt
def predict_api(request):
    if request.method != 'POST' or not request.FILES.get('image'):
        return JsonResponse({'success': False, 'error': 'No image provided.'})

    image_file = request.FILES['image']

    # ── File validation (Fix #13) ────────────────────────────────────────────
    _, ext = os.path.splitext(image_file.name.lower())
    if ext not in ALLOWED_EXTENSIONS:
        return JsonResponse({
            'success': False,
            'error': f'Unsupported file type "{ext}". Please upload JPG, PNG, or DICOM.'
        })

    if image_file.size > MAX_UPLOAD_MB * 1024 * 1024:
        return JsonResponse({
            'success': False,
            'error': f'File too large. Maximum allowed size is {MAX_UPLOAD_MB} MB.'
        })

    # ── Save with UUID prefix to avoid collisions (Fix #6) ──────────────────
    safe_name = uuid.uuid4().hex + ext
    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)

    fs = FileSystemStorage(location=upload_dir)
    saved_name = fs.save(safe_name, image_file)
    input_path = os.path.join(upload_dir, saved_name)

    # ── Build correct media URL manually (Fix #5) ───────────────────────────
    original_url = settings.MEDIA_URL + 'uploads/' + saved_name

    # ── Run inference ────────────────────────────────────────────────────────
    out_dir = os.path.join(settings.MEDIA_ROOT, 'results')
    os.makedirs(out_dir, exist_ok=True)

    try:
        from core_ml.inference_service import get_inference_service
        service = get_inference_service()
        
        # Load custom thresholds from session
        hu_lo = float(request.session.get('hu_lo', 15.0))
        hu_hi = float(request.session.get('hu_hi', 35.0))
        min_area_dicom = int(request.session.get('min_area_dicom', 50))
        min_area_image = int(request.session.get('min_area_image', 30))
        
        result = service.process_image(
            input_path, 
            out_dir,
            hu_lo=hu_lo,
            hu_hi=hu_hi,
            min_area_dicom=min_area_dicom,
            min_area_image=min_area_image
        )
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

    mask_url    = settings.MEDIA_URL + 'results/' + result['mask_filename']
    overlay_url = settings.MEDIA_URL + 'results/' + result['overlay_filename']

    # Convert confidence string (e.g. "36.20%") to float for database storage
    try:
        raw_confidence = float(result['confidence'].replace('%', ''))
    except ValueError:
        raw_confidence = 0.0

    # ── Save to Database ────────────────────────────────────────────────────
    try:
        modality_str = 'NCCT' if ext == '.dcm' else 'Slice Image'
        PatientScan.objects.create(
            scan_name=image_file.name,
            modality=modality_str,
            original_image=original_url,
            mask_image=mask_url,
            overlay_image=overlay_url,
            confidence=raw_confidence,
            detected=result['detected']
        )
    except Exception as db_err:
        # Log error, but don't crash response if db write fails
        print(f"[Database Error] Could not log scan: {db_err}")

    return JsonResponse({
        'success':      True,
        'original_url': original_url,
        'mask_url':     mask_url,
        'overlay_url':  overlay_url,
        'confidence':   result['confidence'],
        'detected':     result['detected'],
    })
