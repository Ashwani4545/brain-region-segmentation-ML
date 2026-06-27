import os
import uuid
import random
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from django.db.models import Avg
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm, PasswordChangeForm
from django.contrib.auth.decorators import login_required
from .models import PatientScan, ChatMessage

# Allowed file extensions for upload validation
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.dcm'}
MAX_UPLOAD_MB = 20

def landing_page(request):
    return render(request, 'segmentation/index.html')

def terms_page(request):
    return render(request, 'segmentation/terms.html')

# ── Authentication Views ──────────────────────────────────────────────────────
def register_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('app')
    else:
        form = UserCreationForm()
    return render(request, 'segmentation/register.html', {'form': form})

def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('app')
    else:
        form = AuthenticationForm()
    return render(request, 'segmentation/login.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('landing')

# ── Protected Views ───────────────────────────────────────────────────────────
@login_required(login_url='login')
def app_page(request):
    return render(request, 'segmentation/app.html')

def about_page(request):
    return render(request, 'segmentation/about.html')

def contact_page(request):
    return render(request, 'segmentation/contact.html')

@login_required(login_url='login')
def dashboard_page(request):
    scans = PatientScan.objects.filter(user=request.user)
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

@login_required(login_url='login')
def profile_page(request):
    user = request.user
    total_scans = PatientScan.objects.filter(user=user).count()
    
    password_form = PasswordChangeForm(user)
    
    if request.method == 'POST':
        if 'update_profile' in request.POST:
            email = request.POST.get('email', '').strip()
            username = request.POST.get('username', '').strip()
            if username:
                user.username = username
            if email:
                user.email = email
            user.save()
            request.session['profile_msg'] = 'Profile updated successfully.'
            return redirect('profile')
            
        elif 'change_password' in request.POST:
            password_form = PasswordChangeForm(user, request.POST)
            if password_form.is_valid():
                user = password_form.save()
                update_session_auth_hash(request, user)  # Keep user logged in
                request.session['profile_msg'] = 'Password changed successfully.'
                return redirect('profile')
            else:
                request.session['profile_err'] = 'Please correct the errors below to change your password.'

    context = {
        'total_scans': total_scans,
        'password_form': password_form,
        'profile_msg': request.session.pop('profile_msg', None),
        'profile_err': request.session.pop('profile_err', None),
    }
    return render(request, 'segmentation/profile.html', context)

@login_required(login_url='login')
def registry_page(request):
    query = request.GET.get('q', '')
    scans = PatientScan.objects.filter(user=request.user)
    if query:
        scans = scans.filter(patient_id__icontains=query) | scans.filter(scan_name__icontains=query)
    
    return render(request, 'segmentation/registry.html', {'scans': scans, 'query': query})

@login_required(login_url='login')
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
    scan_id = None
    try:
        modality_str = 'NCCT' if ext == '.dcm' else 'Slice Image'
        scan = PatientScan.objects.create(
            user=request.user if request.user.is_authenticated else None,
            scan_name=image_file.name,
            modality=modality_str,
            original_image=original_url,
            mask_image=mask_url,
            overlay_image=overlay_url,
            confidence=raw_confidence,
            detected=result['detected']
        )
        scan_id = scan.id
    except Exception as db_err:
        # Log error, but don't crash response if db write fails
        print(f"[Database Error] Could not log scan: {db_err}")

    return JsonResponse({
        'success':      True,
        'scan_id':      scan_id,
        'original_url': original_url,
        'mask_url':     mask_url,
        'overlay_url':  overlay_url,
        'confidence':   result['confidence'],
        'detected':     result['detected'],
    })


@csrf_exempt
def chat_api(request, scan_id):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Method not allowed.'})
    
    try:
        scan = PatientScan.objects.get(id=scan_id)
    except PatientScan.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Scan not found.'})
        
    user_msg_text = request.POST.get('message', '').strip()
    if not user_msg_text:
        return JsonResponse({'success': False, 'error': 'Empty message.'})
        
    # Simple emotion/sentiment check on user message
    emotion_detected = None
    anxious_keywords = ['worried', 'scared', 'afraid', 'dying', 'panic', 'cancer', 'stroke', 'serious', 'fear', 'emergency']
    if any(kw in user_msg_text.lower() for kw in anxious_keywords):
        emotion_detected = 'anxious'
        
    # Save user message to database
    ChatMessage.objects.create(
        scan=scan,
        role='user',
        message=user_msg_text,
        emotion_detected=emotion_detected
    )
    
    # Retrieve chat history (including newly saved user message)
    history_objs = ChatMessage.objects.filter(scan=scan).order_by('timestamp')
    
    # Check if Anthropic API is configured
    api_key = getattr(settings, 'ANTHROPIC_API_KEY', None)
    
    assistant_response = ""
    
    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            
            # Format system prompt
            system_prompt = (
                "You are MedAssist, a compassionate AI healthcare companion on NeuroDetect AI. "
                f"You have just analyzed the patient's Brain {scan.modality} report.\n"
                f"Scan File: {scan.scan_name}\n"
                f"Hypodense regions detected: {scan.detected}\n"
                f"Lesion Load (swelling percentage of brain area): {scan.confidence}%\n\n"
                "Guidelines:\n"
                "1. Speak in simple, comforting, non-medical language.\n"
                "2. Directly acknowledge patient anxiety and validate emotions.\n"
                "3. NEVER give a definitive diagnosis. Reiterate that this is an AI research tool.\n"
                "4. Always recommend consulting a qualified neurologist or radiologist.\n"
                "5. Keep responses concise (under 3-4 paragraphs).\n"
                "6. Never speculate beyond the scan details.\n"
                "7. If severe distress is noted, recommend speaking to family or calling helpline services."
            )
            
            # Format history for Anthropic message list API
            messages = []
            for h in history_objs:
                messages.append({
                    "role": h.role,
                    "content": h.message
                })
                
            # Call Claude
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=800,
                system=system_prompt,
                messages=messages
            )
            
            # Extract content text
            assistant_response = response.content[0].text
            
        except Exception as e:
            # On API failure, fall back to mock
            print(f"[Chat API Error] Anthropic call failed, falling back to mock: {e}")
            assistant_response = get_mock_response(user_msg_text, scan)
    else:
        # Fall back to mock when key is not configured
        assistant_response = get_mock_response(user_msg_text, scan)
        
    # Save assistant response to database
    ChatMessage.objects.create(
        scan=scan,
        role='assistant',
        message=assistant_response
    )
    
    return JsonResponse({
        'success': True,
        'message': assistant_response,
        'emotion_detected': emotion_detected
    })


def get_mock_response(query, scan):
    q = query.lower()
    lesion_str = f"{scan.confidence:.2f}%"
    detected_status = "hypodense regions have been identified" if scan.detected else "no significant hypodense regions were detected"
    
    # 1. Emergency detection
    if any(w in q for w in ['emergency', 'dying', 'chest pain', 'numbness', 'drooping', 'stroke right now']):
        return (
            "I understand this is incredibly frightening, but please stay as calm as possible. If you or the patient are experiencing "
            "active stroke symptoms—such as facial drooping, arm weakness, or speech difficulties (FAST)—please call emergency medical services "
            "immediately (like 112 or 102/108 in India). Time is critical in these situations, and a hospital ER is the safest place to be."
        )
        
    # 2. Lesion load / Swelling
    if any(w in q for w in ['lesion', 'load', 'percentage', 'swelling', 'size']):
        return (
            f"The 'lesion load' of {lesion_str} indicates the portion of the brain's soft tissue area that shows decreased density "
            f"(hypodensity) compared to normal tissue. On a CT scan, this is typically where cytotoxic edema (swelling) might be occurring. "
            f"While {lesion_str} is the estimated region size, only a specialized neuroradiologist can confirm if this corresponds to an "
            f"actual lesion or normal anatomical variation. I highly recommend taking this report to your doctor."
        )
        
    # 3. What is stroke / hypodensity
    if any(w in q for w in ['stroke', 'hypodense', 'hypodensity', 'infarct', 'ischemic']):
        return (
            "Hypodensity refers to areas on a Brain CT scan that appear darker than normal brain tissue. Darker regions can develop when "
            "brain tissue absorbs water, which is a common early response to reduced blood flow (ischaemic changes or cytotoxic edema). "
            "While the AI model flagged these areas, other conditions can also cause hypodensities. A neurologist will correlate this "
            "with clinical symptoms and likely order a follow-up MRI to get a clearer picture."
        )
        
    # 4. What doctor / specialist
    if any(w in q for w in ['doctor', 'specialist', 'neurologist', 'radiologist', 'hospital', 'see']):
        return (
            "You should consult a **Neurologist** or a **Neuroradiologist** as soon as possible. They are the medical specialists trained "
            "to read brain scans and diagnose neurological conditions. When you see them, you should ask:\n"
            "1. Does this hypodensity correspond to an acute ischemic change?\n"
            "2. Should we perform a follow-up DWI-MRI scan?\n"
            "3. Are there signs of edema or mass effect that require immediate treatment?"
        )
        
    # 5. Anxiety reassurance
    if any(w in q for w in ['worried', 'scared', 'afraid', 'panic', 'fear', 'anxious']):
        return (
            "It is completely normal to feel scared and anxious when looking at a brain report. Please remember that this AI output is "
            "an academic research tool and not a final medical diagnosis. A positive flag simply means 'please look closer'. Many dark spots "
            "on CT scans turn out to be completely benign or older, stable changes. Take a deep breath, and let's work on scheduling a "
            "consultation with a specialist to review these findings together."
        )
        
    # Default fallback greeting or Q&A response
    return (
        f"Regarding your query, let's review the scan findings: the AI model processed {scan.scan_name} and noted that {detected_status} "
        f"with an estimated tissue involvement of {lesion_str}. Please consult a neurologist to correlate these findings with "
        f"any clinical symptoms. Let me know if you would like me to explain what hypodensity means, recommend specialists, or generate "
        f"questions you can ask your doctor."
    )
