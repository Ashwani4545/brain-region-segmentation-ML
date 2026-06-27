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
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.dcm', '.pdf'}
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
            'error': f'Unsupported file type "{ext}". Please upload JPG, PNG, DICOM, or PDF.'
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

    # ── Run Ingestion & Analysis Router ──────────────────────────────────────
    out_dir = os.path.join(settings.MEDIA_ROOT, 'results')
    os.makedirs(out_dir, exist_ok=True)

    try:
        from core_ml.ingestion import get_ingestion_service
        ingestion = get_ingestion_service()
        
        # Route file to correct analyzer (Brain CT, Chest X-ray, ECG, or Blood Test)
        routed = ingestion.route_file(input_path, out_dir)
        modality = routed['modality']
        result = routed['result']
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

    # Determine mask and overlay URLs
    if modality == 'BLOOD_TEST':
        mask_url = ""
        overlay_url = ""
    else:
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
        scan = PatientScan.objects.create(
            user=request.user if request.user.is_authenticated else None,
            scan_name=image_file.name,
            modality=modality,
            original_image=original_url,
            mask_image=mask_url,
            overlay_image=overlay_url,
            confidence=raw_confidence,
            detected=result['detected'],
            notes=result.get('findings_text', '') # Store raw clinical findings in notes
        )
        scan_id = scan.id
    except Exception as db_err:
        # Log error, but don't crash response if db write fails
        print(f"[Database Error] Could not log scan: {db_err}")

    return JsonResponse({
        'success':      True,
        'scan_id':      scan_id,
        'modality':     modality,
        'original_url': original_url,
        'mask_url':     mask_url,
        'overlay_url':  overlay_url,
        'confidence':   result['confidence'],
        'detected':     result['detected'],
        'findings_text': result.get('findings_text', ''),
        'metrics':       result.get('metrics', []) # blood test parsed parameters
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
            
            # Determine specialist, findings label, and custom guidelines based on modality
            specialist = "neurologist"
            findings_label = "Hypodense regions"
            extra_instructions = ""
            
            if scan.modality == 'CXR':
                specialist = "pulmonologist"
                findings_label = "Lung anomalies / consolidation"
                extra_instructions = "Explain chest X-ray findings. Guide the patient regarding respiratory safety and symptoms like cough or shortness of breath."
            elif scan.modality == 'ECG':
                specialist = "cardiologist"
                findings_label = "Heart rhythm abnormalities"
                extra_instructions = "Explain Heart Rate (BPM) and standard waveform traces, and warn about symptoms like chest pain."
            elif scan.modality == 'BLOOD_TEST':
                specialist = "general physician"
                findings_label = "Out-of-range lab metrics"
                extra_instructions = "Help interpret metabolic, hematology, or sugar levels, and suggest standard diet tips."
            else:
                extra_instructions = "Explain brain CT scans. Warn about stroke signs like facial drooping, arm weakness, or slurred speech."
                
            # Format system prompt
            system_prompt = (
                "You are MedAssist, a compassionate AI healthcare companion on NeuroDetect AI. "
                f"You have just analyzed the patient's {scan.modality} report.\n"
                f"Scan File: {scan.scan_name}\n"
                f"Anomalies detected ({findings_label}): {scan.detected}\n"
                f"Telemetry Metric (Confidence/Percentage): {scan.confidence}%\n"
                f"Raw Findings Details: {scan.notes or ''}\n\n"
                "Guidelines:\n"
                "1. Speak in simple, comforting, non-medical language.\n"
                "2. Directly acknowledge patient anxiety and validate emotions.\n"
                "3. NEVER give a definitive diagnosis. Reiterate that this is an AI research tool.\n"
                f"4. Always recommend consulting a qualified {specialist}.\n"
                "5. Keep responses concise (under 3-4 paragraphs).\n"
                "6. Never speculate beyond the provided details.\n"
                f"7. {extra_instructions}\n"
                "8. If severe distress is noted, recommend speaking to family or calling emergency medical services."
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
    m = scan.modality
    val_str = f"{scan.confidence:.2f}%"
    detected_status = "anomalies have been identified" if scan.detected else "no significant anomalies were detected"
    
    # Common Emergency check
    if any(w in q for w in ['emergency', 'dying', 'chest pain', 'severe pain', 'numbness', 'drooping', 'stroke right now']):
        if m == 'ECG' or m == 'CXR':
            return (
                "Please stay calm but act immediately. If you or the patient are experiencing chest pain, severe shortness of breath, "
                "or radiative left-arm numbness, please call emergency services (like 112 or 102/108 in India, or 911) immediately. "
                "These symptoms require instant evaluation in a hospital emergency room."
            )
        elif m == 'CT':
            return (
                "If you or the patient are experiencing active stroke symptoms—such as facial drooping, arm weakness, or speech difficulties (FAST)—"
                "please call emergency medical services immediately. Time is critical, and a hospital ER is the safest place to be."
            )
        else:
            return (
                "If you are experiencing severe, sudden symptoms, please contact emergency services immediately. An AI tool cannot "
                "evaluate acute or life-threatening crises."
            )
            
    # Modality-specific mock QA
    if m == 'CT':
        if any(w in q for w in ['lesion', 'load', 'percentage', 'swelling', 'size']):
            return (
                f"The 'lesion load' of {val_str} indicates the portion of the brain's soft tissue area that shows decreased density "
                f"(hypodensity) compared to normal tissue. On a CT scan, this is typically where cytotoxic edema (swelling) might be occurring. "
                f"While {val_str} is the estimated region size, only a specialized neuroradiologist can confirm if this corresponds to an "
                f"actual lesion or normal anatomical variation. I highly recommend taking this report to your doctor."
            )
        if any(w in q for w in ['stroke', 'hypodense', 'hypodensity', 'infarct', 'ischemic']):
            return (
                "Hypodensity refers to areas on a Brain CT scan that appear darker than normal brain tissue. Darker regions can develop when "
                "brain tissue absorbs water, which is a common early response to reduced blood flow (ischaemic changes or cytotoxic edema). "
                "While the AI model flagged these areas, other conditions can also cause hypodensities. A neurologist will correlate this "
                "with clinical symptoms and likely order a follow-up MRI to get a clearer picture."
            )
        if any(w in q for w in ['doctor', 'specialist', 'neurologist', 'radiologist', 'hospital', 'see']):
            return (
                "You should consult a **Neurologist** or a **Neuroradiologist** as soon as possible. They are the medical specialists trained "
                "to read brain scans and diagnose neurological conditions. When you see them, you should ask:\n"
                "1. Does this hypodensity correspond to an acute ischemic change?\n"
                "2. Should we perform a follow-up DWI-MRI scan?\n"
                "3. Are there signs of edema or mass effect that require immediate treatment?"
            )
            
    elif m == 'CXR':
        if any(w in q for w in ['lung', 'pneumonia', 'consolidation', 'opacity', 'fluid']):
            return (
                "In a Chest X-ray, normally black regions (air-filled lungs) that appear white or opaque indicate 'consolidation' or fluid accumulation. "
                "This is a common marker for pneumonia, infections, or inflammation. The AI model checks for these densities. "
                "If opacity is found, a pulmonologist will correlate it with symptoms like fever, cough, and oxygen levels to diagnose."
            )
        if any(w in q for w in ['heart', 'cardiomegaly', 'enlarged', 'ctr', 'ratio']):
            return (
                f"Your Cardiothoracic Ratio (CTR) was calculated. A ratio greater than 0.50 (50%) indicates 'cardiomegaly' or an "
                f"enlarged heart silhouette. This can be caused by chronic high blood pressure, valve conditions, or heart failure. "
                f"A cardiologist can confirm this with a simple echocardiogram."
            )
        if any(w in q for w in ['doctor', 'specialist', 'pulmonologist', 'cardiologist', 'see']):
            return (
                "You should consult a **Pulmonologist** (for lung opacities) or a **Cardiologist** (for heart shadow concerns). "
                "Important questions to ask them include:\n"
                "1. Does this consolidation point to bacterial pneumonia or another source?\n"
                "2. Do I need an echo to evaluate the cardiomegaly?\n"
                "3. Should we prescribe antibiotics or order a chest CT for higher resolution?"
            )
            
    elif m == 'ECG':
        if any(w in q for w in ['rate', 'bpm', 'heartbeat', 'tachycardia', 'bradycardia']):
            return (
                f"Your estimated heart rate is {scan.confidence} BPM. A resting rate above 100 BPM is classified as tachycardia (fast heart rate), "
                f"while a resting rate below 60 BPM is bradycardia (slow heart rate). Factors like anxiety, dehydration, medications, or "
                f"rhythm disturbances can alter BPM. A cardiologist will evaluate if this heartbeat represents a normal sinus rhythm."
            )
        if any(w in q for w in ['rhythm', 'arrhythmia', 'irregular', 'peak', 'pulse']):
            return (
                "Rhythms are irregular when the time interval between consecutive heartbeat peaks (R-R intervals) varies. "
                "This can indicate sinus arrhythmia (harmless variation with breathing), premature beats (PVCs), or conditions "
                "like atrial fibrillation. A cardiologist will verify this with a 12-lead ECG or a 24-hour Holter monitor."
            )
        if any(w in q for w in ['doctor', 'specialist', 'cardiologist', 'see']):
            return (
                "You should consult a **Cardiologist**. They are the heart specialists. Key questions to ask include:\n"
                "1. Does this tracing show signs of ischemia, blockages, or arrhythmia?\n"
                "2. Do I need a 24-hour Holter monitor or stress test?\n"
                "3. What lifestyle adjustments or medications do you recommend for this rhythm?"
            )
            
    elif m == 'BLOOD_TEST':
        if any(w in q for w in ['metric', 'range', 'low', 'high', 'out', 'normal']):
            return (
                "Out-of-range lab metrics indicate values that fall below the minimum or above the maximum bounds defined for healthy reference groups. "
                "For example, low hemoglobin suggests anemia, while high fasting glucose indicates hyperglycemia. "
                "A physician will evaluate these numbers together with your hydration level, diet, and clinical history."
            )
        if any(w in q for w in ['sugar', 'glucose', 'diabetes', 'sweet', 'carb']):
            return (
                "Glucose measures the sugar concentration in your blood. Fasting values above 100 mg/dL suggest pre-diabetes, and values "
                "above 126 mg/dL suggest diabetes. You should consult a physician for diagnostic confirmation and discuss a diet low in "
                "refined carbohydrates."
            )
        if any(w in q for w in ['hemoglobin', 'hb', 'anemia', 'blood', 'iron']):
            return (
                "Hemoglobin is the iron-rich protein in red blood cells that carries oxygen to your body's tissues. Values below 12.0 g/dL "
                "suggest anemia, which can cause fatigue, weakness, or cold hands. A physician can check if this is iron-deficiency anemia "
                "and recommend dietary changes or iron supplements."
            )
        if any(w in q for w in ['doctor', 'specialist', 'physician', 'see']):
            return (
                "You should consult a **General Physician** or **Endocrinologist**. Key questions to ask them:\n"
                "1. Do these out-of-range parameters require medical treatment or simple dietary changes?\n"
                "2. Should we repeat this blood draw to verify levels?\n"
                "3. Do I need additional checks, like HbA1c for glucose or ferritin for iron?"
            )

    # General / Fallback reassurance QA
    if any(w in q for w in ['worried', 'scared', 'afraid', 'panic', 'fear', 'anxious']):
        return (
            "It is completely normal to feel worried when reviewing clinical results. Please remember that this AI analysis "
            "is a research tool and not a final medical diagnosis. A flag simply suggests 'please look closer'. Many flags turn out to "
            "be stable, benign, or normal variations. Take a deep breath and let's work on scheduling a professional review with a doctor."
        )

    # Default fallback greeting
    return (
        f"Regarding your query on this {m} report: the AI router has analyzed {scan.scan_name} and noted that {detected_status}. "
        f"I highly recommend consulting a specialist to correlate this with clinical symptoms. Let me know if you would like me "
        f"to explain specific terms, recommend doctors, or generate consult questions."
    )
