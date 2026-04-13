import os
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage

# Setup core ml path 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from core_ml.inference_service import get_inference_service
except ImportError:
    pass

def landing_page(request):
    return render(request, 'segmentation/index.html')

def app_page(request):
    return render(request, 'segmentation/app.html')

@csrf_exempt
def predict_api(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        
        # Save uploaded file
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        fs = FileSystemStorage(location=upload_dir)
        filename = fs.save(image_file.name, image_file)
        file_url = fs.url(filename)
        input_path = os.path.join(upload_dir, filename)
        
        # Create output dir
        out_dir = os.path.join(settings.MEDIA_ROOT, 'results')
        os.makedirs(out_dir, exist_ok=True)
        
        try:
            service = get_inference_service()
            result = service.process_image(input_path, out_dir)
            
            # Format URLs
            mask_url = settings.MEDIA_URL + 'results/' + result['mask_filename']
            overlay_url = settings.MEDIA_URL + 'results/' + result['overlay_filename']
            
            return JsonResponse({
                'success': True,
                'original_url': settings.MEDIA_URL + 'uploads/' + filename,
                'mask_url': mask_url,
                'overlay_url': overlay_url,
                'confidence': result['confidence'],
                'detected': result['detected']
            })
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})

    return JsonResponse({'success': False, 'error': 'No image provided.'})
