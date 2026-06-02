import os
import uuid

from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage


# Allowed file extensions for upload validation
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.dcm'}
MAX_UPLOAD_MB = 20


def landing_page(request):
    return render(request, 'segmentation/index.html')


def app_page(request):
    return render(request, 'segmentation/app.html')


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
        result = service.process_image(input_path, out_dir)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

    mask_url    = settings.MEDIA_URL + 'results/' + result['mask_filename']
    overlay_url = settings.MEDIA_URL + 'results/' + result['overlay_filename']

    return JsonResponse({
        'success':      True,
        'original_url': original_url,
        'mask_url':     mask_url,
        'overlay_url':  overlay_url,
        'confidence':   result['confidence'],
        'detected':     result['detected'],
    })
