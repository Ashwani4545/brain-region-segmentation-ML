document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const loadingState = document.getElementById('loadingState');
    const resultsArea = document.getElementById('resultsArea');
    const resetBtn = document.getElementById('resetBtn');

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    // Highlight drop area
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });

    // Handle dropped files
    uploadArea.addEventListener('drop', handleDrop, false);
    
    // Handle click to browse
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            uploadFile(this.files[0]);
        }
    });

    let currentScanId = null;
    let currentConsultId = null;
    let telehealthPollInterval = null;
    let typingIndicatorElem = null;

    // Setup Tab Switcher Handlers
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');
    
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            
            tabButtons.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(p => {
                p.classList.add('hidden');
                p.classList.remove('active');
            });
            
            btn.classList.add('active');
            const targetPane = document.getElementById(`tab-${targetTab}`);
            if (targetPane) {
                targetPane.classList.remove('hidden');
                targetPane.classList.add('active');
            }

            // Clear any active telehealth polling
            if (telehealthPollInterval) {
                clearInterval(telehealthPollInterval);
                telehealthPollInterval = null;
            }

            // Telehealth room activation
            if (targetTab === 'telehealth') {
                if (currentConsultId) {
                    loadConsultationMessages();
                    telehealthPollInterval = setInterval(loadConsultationMessages, 4000);
                } else if (currentScanId) {
                    fetchNearbySpecialists();
                }
            }
        });
    });

    // Reset button handler
    resetBtn.addEventListener('click', () => {
        resultsArea.classList.add('hidden');
        uploadArea.classList.remove('hidden');
        fileInput.value = '';
        currentScanId = null;
        currentConsultId = null;
        if (telehealthPollInterval) {
            clearInterval(telehealthPollInterval);
            telehealthPollInterval = null;
        }
        document.getElementById('chatMessages').innerHTML = '';
        
        // Reset view states
        document.getElementById('telehealthReferralView').classList.remove('hidden');
        document.getElementById('telehealthActiveRoom').classList.add('hidden');
        
        // Reset tabs to default active
        tabButtons.forEach(b => b.classList.remove('active'));
        tabPanes.forEach(p => {
            p.classList.add('hidden');
            p.classList.remove('active');
        });
        const firstBtn = document.querySelector('.tab-btn[data-tab="imaging"]');
        if (firstBtn) firstBtn.classList.add('active');
        const firstPane = document.getElementById('tab-imaging');
        if (firstPane) {
            firstPane.classList.remove('hidden');
            firstPane.classList.add('active');
        }
    });

    // Chat elements
    const chatInput = document.getElementById('chatInput');
    const sendChatBtn = document.getElementById('sendChatBtn');
    const chatTags = document.querySelectorAll('.chat-tag');

    if (chatInput && sendChatBtn) {
        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                sendChatMessage(chatInput.value);
            }
        });
        sendChatBtn.addEventListener('click', () => {
            sendChatMessage(chatInput.value);
        });
    }

    chatTags.forEach(tag => {
        tag.addEventListener('click', function() {
            const msg = this.getAttribute('data-msg');
            if (msg) {
                sendChatMessage(msg);
            }
        });
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight(e) {
        uploadArea.classList.add('dragover');
    }

    function unhighlight(e) {
        uploadArea.classList.remove('dragover');
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const file = dt.files[0];
        if (file) uploadFile(file);
    }

    function uploadFile(file) {
        // UI transitions
        uploadArea.classList.add('hidden');
        loadingState.classList.remove('hidden');
        resultsArea.classList.add('hidden');
        currentScanId = null;

        const formData = new FormData();
        formData.append('image', file);

        // Fetch CSRF Token
        const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]')?.value || '';
        formData.append('csrfmiddlewaretoken', csrfToken);

        fetch(predictApiUrl, {
            method: 'POST',
            body: formData,
            headers: {
                'X-CSRFToken': csrfToken
            }
        })
        .then(response => response.json())
        .then(data => {
            loadingState.classList.add('hidden');
            
            if (data.success) {
                currentScanId = data.scan_id;
                
                // Toggle display widgets based on modality
                const imageContainer = document.getElementById('imageResultsContainer');
                const bloodContainer = document.getElementById('bloodResultsContainer');
                imageContainer.classList.add('hidden');
                bloodContainer.classList.add('hidden');
                
                const labelOrig = document.getElementById('labelOrig');
                const labelMask = document.getElementById('labelMask');
                const labelOverlay = document.getElementById('labelOverlay');
                const maskBox = document.getElementById('maskBox');
                const overlayBox = document.getElementById('overlayBox');
                
                // Reset labels
                labelOrig.innerText = "Original Scan";
                labelMask.innerText = "Segmented Mask";
                labelOverlay.innerHTML = `AI Overlay <span class="med-badge med-badge-success" style="vertical-align: 1px; margin-left: 5px;">Annotated</span>`;
                maskBox.classList.remove('hidden');
                overlayBox.classList.remove('hidden');
                
                if (data.modality === 'BLOOD_TEST') {
                    // Render blood table
                    bloodContainer.classList.remove('hidden');
                    const tbody = document.getElementById('bloodMetricsBody');
                    tbody.innerHTML = '';
                    
                    data.metrics.forEach(m => {
                        const tr = document.createElement('tr');
                        let badgeClass = 'med-badge-success';
                        if (m.status === 'Low') badgeClass = 'med-badge-warning';
                        if (m.status === 'High') badgeClass = 'med-badge-danger';
                        
                        tr.innerHTML = `
                            <td style="font-weight: 600;">${m.marker}</td>
                            <td>${m.value} ${m.unit}</td>
                            <td>${m.min} - ${m.max} ${m.unit}</td>
                            <td><span class="med-badge ${badgeClass}">${m.status}</span></td>
                        `;
                        tbody.appendChild(tr);
                    });
                } else {
                    // Render image grid
                    imageContainer.classList.remove('hidden');
                    
                    if (data.modality === 'CXR') {
                        labelOrig.innerText = "Original X-Ray";
                        labelMask.innerText = "Segmented Consolidation";
                        labelOverlay.innerHTML = `Chest Overlay <span class="med-badge med-badge-success" style="vertical-align: 1px; margin-left: 5px;">Lobe Annotation</span>`;
                    } else if (data.modality === 'ECG') {
                        labelOrig.innerText = "Original ECG Trace";
                        labelMask.innerText = "Extracted Signal Line";
                        labelOverlay.innerHTML = `Waveform Analysis <span class="med-badge med-badge-success" style="vertical-align: 1px; margin-left: 5px;">R-Peak Detections</span>`;
                    }
                    
                    document.getElementById('origImage').src = data.original_url;
                    document.getElementById('maskImage').src = data.mask_url;
                    document.getElementById('overlayImage').src = data.overlay_url;
                    
                    document.getElementById('downloadBtn').href = data.overlay_url;
                }
                
                // Populate confidence badge
                const badge = document.getElementById('confidenceBadge');
                badge.innerText = `Confidence: ${data.confidence}`;
                
                if (data.detected) {
                    badge.style.color = '#ef4444'; // Red showing danger
                    badge.style.borderColor = 'rgba(239, 68, 68, 0.2)';
                    badge.style.background = 'rgba(239, 68, 68, 0.1)';
                    
                    if (data.modality === 'BLOOD_TEST') {
                        document.getElementById('explainText').innerText = `Anomaly Detected. The blood panel indicates parameters that fall outside standard physiological reference ranges. Findings: ${data.findings_text}`;
                    } else if (data.modality === 'CXR') {
                        document.getElementById('explainText').innerText = `Anomaly Detected. The chest X-ray indicates consolidation or cardiovascular silhouettes. Findings: ${data.findings_text}`;
                    } else if (data.modality === 'ECG') {
                        document.getElementById('explainText').innerText = `Anomaly Detected. The ECG trace indicates heart rate or rhythm variances. Findings: ${data.findings_text}`;
                    } else {
                        document.getElementById('explainText').innerText = `Anomaly Detected. The system detected hypodense regions indicating possible stroke-affected areas. Confidence level: ${data.confidence}. See overlay for exact visual localization.`;
                    }
                } else {
                    badge.style.color = '#10b981'; // Green showing clean
                    badge.style.borderColor = 'rgba(16, 185, 129, 0.2)';
                    badge.style.background = 'rgba(16, 185, 129, 0.1)';
                    
                    if (data.modality === 'BLOOD_TEST') {
                        document.getElementById('explainText').innerText = `No Analysis Findings. All blood test metrics are within reference ranges.`;
                    } else {
                        document.getElementById('explainText').innerText = `No Analysis Findings. The model did not detect significant anomalies in this scan slice.`;
                    }
                }

                // Populate interactive RAG recovery tabs
                populateGuidanceTabs(data.guidance);

                // Populate predictive risk profiles
                populateRiskProfiler(data.risk_profile);

                // Populate telehealth referral specialists
                updateTelehealthReferrals(data.specialists);

                // Load patient history timeline
                updatePatientHistoryTimeline();

                resultsArea.classList.remove('hidden');

                // Dynamic tags loading per modality
                const tagsContainer = document.getElementById('chatTags');
                if (data.modality === 'BLOOD_TEST') {
                    tagsContainer.innerHTML = `
                        <span class="chat-tag" data-msg="Explain what low hemoglobin means.">Low Hemoglobin?</span>
                        <span class="chat-tag" data-msg="What are the implications of high fasting glucose?">High Blood Glucose?</span>
                        <span class="chat-tag" data-msg="What general doctor should I see to review these lab results?">Which Doctor?</span>
                        <span class="chat-tag" data-msg="Are there standard dietary changes for high cholesterol?">Diet Tips</span>
                    `;
                } else if (data.modality === 'CXR') {
                    tagsContainer.innerHTML = `
                        <span class="chat-tag" data-msg="What is lung consolidation/opacity?">Lung Opacity?</span>
                        <span class="chat-tag" data-msg="What does an enlarged heart CTR mean?">Enlarged Heart?</span>
                        <span class="chat-tag" data-msg="What doctor should I consult for these lung opacities?">Consult Specialist</span>
                        <span class="chat-tag" data-msg="Is lung consolidation an emergency?">Emergency signs?</span>
                    `;
                } else if (data.modality === 'ECG') {
                    tagsContainer.innerHTML = `
                        <span class="chat-tag" data-msg="Explain my Heart Rate BPM results.">Heart Rate BPM?</span>
                        <span class="chat-tag" data-msg="What does an arrhythmia or irregular R-R mean?">Arrhythmia?</span>
                        <span class="chat-tag" data-msg="What cardiologist should I consult?">Consult Cardiologist</span>
                        <span class="chat-tag" data-msg="Is an irregular ECG tracing dangerous?">Is it dangerous?</span>
                    `;
                } else {
                    tagsContainer.innerHTML = `
                        <span class="chat-tag" data-msg="Explain the scan findings in simple terms.">Explain Scan</span>
                        <span class="chat-tag" data-msg="What does lesion load indicate?">What is Lesion Load?</span>
                        <span class="chat-tag" data-msg="What doctor should I see next?">Specialist Referral</span>
                        <span class="chat-tag" data-msg="Is this result an emergency?">Emergency Signs?</span>
                    `;
                }
                
                // Re-bind events to the new tags
                document.querySelectorAll('.chat-tag').forEach(tag => {
                    tag.addEventListener('click', function() {
                        const msg = this.getAttribute('data-msg');
                        if (msg) {
                            sendChatMessage(msg);
                        }
                    });
                });

                // Clear old chat messages and disable/enable controls
                const chatMessages = document.getElementById('chatMessages');
                chatMessages.innerHTML = '';
                chatInput.disabled = false;
                sendChatBtn.disabled = false;

                // Auto-trigger MedAssist welcome message after 2 seconds
                setTimeout(() => {
                    triggerWelcomeMessage(data);
                }, 2000);

            } else {
                alert('Error processing image: ' + data.error);
                uploadArea.classList.remove('hidden');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            loadingState.classList.add('hidden');
            uploadArea.classList.remove('hidden');
            alert('A network error occurred.');
        });
    }

    function triggerWelcomeMessage(data) {
        let modalityName = "Brain CT";
        if (data.modality === 'CXR') modalityName = "Chest X-Ray";
        if (data.modality === 'ECG') modalityName = "ECG Waveform";
        if (data.modality === 'BLOOD_TEST') modalityName = "Blood Panel";
        
        const msg = data.detected 
            ? `Hello, I am MedAssist, your compassionate care companion. I have reviewed your ${modalityName} results. The analysis noted some anomalies (${data.findings_text}). I understand this can cause anxiety, but please know I am here to help explain what these results mean in simple terms. How can I support you right now?`
            : `Hello, I am MedAssist, your compassionate care companion. I have reviewed your ${modalityName} report. The analysis did not flag any significant anomalies (confidence: ${data.confidence}). I am here if you have any questions or would like details on any of these parameters. What would you like to discuss?`;
            
        addMessageBubble('assistant', msg);
    }

    function addMessageBubble(role, text) {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return;

        const bubble = document.createElement('div');
        bubble.className = `chat-bubble ${role}`;
        
        // Handle line breaks and lists
        if (text.includes('\n')) {
            const lines = text.split('\n');
            lines.forEach(line => {
                const p = document.createElement('p');
                p.innerText = line;
                bubble.appendChild(p);
            });
        } else {
            bubble.innerText = text;
        }
        
        chatMessages.appendChild(bubble);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function showTypingIndicator() {
        if (typingIndicatorElem) return;
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return;

        const indicator = document.createElement('div');
        indicator.className = 'chat-bubble assistant';
        indicator.id = 'typingIndicator';
        indicator.innerHTML = `
            <div class="typing-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        `;
        chatMessages.appendChild(indicator);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        typingIndicatorElem = indicator;
    }

    function hideTypingIndicator() {
        if (typingIndicatorElem) {
            typingIndicatorElem.remove();
            typingIndicatorElem = null;
        }
    }

    function sendChatMessage(text) {
        if (!text.trim() || !currentScanId) return;
        
        addMessageBubble('user', text);
        showTypingIndicator();
        
        // Clear input
        chatInput.value = '';
        
        const formData = new FormData();
        formData.append('message', text);
        const langSelect = document.getElementById('chatLanguageSelect');
        const lang = langSelect ? langSelect.value : 'English';
        formData.append('language', lang);
        
        const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]')?.value || '';
        formData.append('csrfmiddlewaretoken', csrfToken);
        
        fetch(`/api/chat/${currentScanId}/`, {
            method: 'POST',
            body: formData,
            headers: {
                'X-CSRFToken': csrfToken
            }
        })
        .then(response => response.json())
        .then(data => {
            hideTypingIndicator();
            if (data.success) {
                addMessageBubble('assistant', data.message);
            } else {
                addMessageBubble('assistant', "I apologize, but I encountered an error processing your query. Please try again.");
            }
        })
        .catch(error => {
            hideTypingIndicator();
            console.error('Chat error:', error);
            addMessageBubble('assistant', "A network error occurred. Please verify your connection.");
        });
    }

    function populateGuidanceTabs(guidance) {
        if (!guidance) return;
        
        // 1. Populate Diet Tab
        const diet = guidance.diet || {};
        const recList = document.getElementById('dietRecommendedList');
        const avoidList = document.getElementById('dietAvoidList');
        recList.innerHTML = '';
        avoidList.innerHTML = '';
        
        (diet.recommended || []).forEach(item => {
            const li = document.createElement('li');
            li.innerText = item;
            recList.appendChild(li);
        });
        (diet.avoid || []).forEach(item => {
            const li = document.createElement('li');
            li.innerText = item;
            avoidList.appendChild(li);
        });
        document.getElementById('dietReasoningText').innerText = diet.reasoning || '--';
        
        // 2. Populate Recovery & Exercise Tab
        const exercise = guidance.exercise || {};
        const allowedList = document.getElementById('exerciseAllowedList');
        const restrictedList = document.getElementById('exerciseRestrictedList');
        allowedList.innerHTML = '';
        restrictedList.innerHTML = '';
        
        (exercise.allowed || []).forEach(item => {
            const li = document.createElement('li');
            li.innerText = item;
            allowedList.appendChild(li);
        });
        (exercise.restrictions || []).forEach(item => {
            const li = document.createElement('li');
            li.innerText = item;
            restrictedList.appendChild(li);
        });
        
        const lifestyle = guidance.lifestyle || {};
        document.getElementById('lifestyleNotesText').innerText = lifestyle.notes || '--';
        
        // 3. Populate Doctor Guide Tab
        const routing = guidance.routing || {};
        document.getElementById('referralSpecialist').innerText = routing.specialist || '--';
        
        const urgencyBadge = document.getElementById('referralUrgency');
        urgencyBadge.innerText = routing.urgency || '--';
        urgencyBadge.className = 'med-badge';
        
        const urg = (routing.urgency || '').toLowerCase();
        if (urg.includes('immediate') || urg.includes('high')) {
            urgencyBadge.classList.add('med-badge-danger');
        } else if (urg.includes('moderate')) {
            urgencyBadge.classList.add('med-badge-warning');
        } else {
            urgencyBadge.classList.add('med-badge-success');
        }
        
        document.getElementById('referralGuidance').innerText = routing.guidance || '--';
        
        const questionsList = document.getElementById('doctorQuestionsList');
        questionsList.innerHTML = '';
        (guidance.questions || []).forEach(item => {
            const li = document.createElement('li');
            li.innerText = item;
            li.style.marginBottom = '4px';
            questionsList.appendChild(li);
        });
    }

    // Setup Risk Profiler form handler
    const riskOverrideForm = document.getElementById('riskOverrideForm');
    if (riskOverrideForm) {
        riskOverrideForm.addEventListener('submit', (e) => {
            e.preventDefault();
            if (!currentScanId) return;
            
            const submitBtn = riskOverrideForm.querySelector('button[type="submit"]');
            const originalBtnHtml = submitBtn.innerHTML;
            submitBtn.disabled = true;
            submitBtn.innerHTML = `<span class="spinner-sm" style="display: inline-block; width: 10px; height: 10px; border: 2px solid #fff; border-top: 2px solid transparent; border-radius: 50%; animation: spin 1s linear infinite; margin-right: 4px;"></span> Recalculating...`;
            
            const formData = new FormData(riskOverrideForm);
            
            fetch(`/api/scan/${currentScanId}/recalculate_risk/`, {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                submitBtn.disabled = false;
                submitBtn.innerHTML = originalBtnHtml;
                if (data.success) {
                    populateRiskProfiler(data.risk_profile);
                } else {
                    alert('Error recalculating risk profile: ' + data.error);
                }
            })
            .catch(error => {
                submitBtn.disabled = false;
                submitBtn.innerHTML = originalBtnHtml;
                console.error('Recalculate error:', error);
                alert('A network error occurred.');
            });
        });
    }

    function populateRiskProfiler(profile) {
        if (!profile) return;
        
        // 1. Set form input default values from active profile data
        const metrics = profile.detailed_metrics || {};
        if (document.getElementById('riskAgeInput')) document.getElementById('riskAgeInput').value = metrics.age || 52;
        if (document.getElementById('riskBpInput')) document.getElementById('riskBpInput').value = metrics.systolic_bp || 128;
        if (document.getElementById('riskBmiInput')) document.getElementById('riskBmiInput').value = metrics.bmi || 24.2;
        if (document.getElementById('riskSmokingInput')) document.getElementById('riskSmokingInput').value = metrics.smoking || 'No';
        
        // 2. Animate and colorize SVG Gauges
        animateRiskGauge('cvGaugeStroke', 'cvRiskScoreText', 'cvRiskBadge', profile.cv_risk_score, profile.cv_risk_grade);
        animateRiskGauge('dbGaugeStroke', 'dbRiskScoreText', 'dbRiskBadge', profile.diabetes_risk_score, profile.diabetes_risk_grade);
        animateRiskGauge('strokeGaugeStroke', 'strokeRiskScoreText', 'strokeRiskBadge', profile.stroke_risk_score, profile.stroke_risk_grade);
        
        // 3. Populate Explanation Card
        const cvTxt = `Cardiovascular risk is ${profile.cv_risk_grade} (${profile.cv_risk_score}%).`;
        const dbTxt = `Diabetes Type-II risk is ${profile.diabetes_risk_grade} (${profile.diabetes_risk_score}%).`;
        const stTxt = `Stroke risk is ${profile.stroke_risk_grade} (${profile.stroke_risk_score}%).`;
        
        document.getElementById('riskExplanationText').innerHTML = `
            <strong style="color: var(--text-primary);">Clinical Profile Evaluation:</strong><br>
            • ${cvTxt} based on age, blood pressure, and cholesterol markers.<br>
            • ${dbTxt} reflecting blood glucose and body composition levels.<br>
            • ${stTxt} incorporating CT imaging scans and vascular dynamics.<br><br>
            <em style="font-size: 11px; color: var(--text-secondary);">Recalculate parameters on the left to test individual risk variables. All indicators are estimations for proactive monitoring and do not replace formal clinical assessment.</em>
        `;
    }

    function animateRiskGauge(gaugeStrokeId, scoreTextId, badgeId, percent, grade) {
        const strokePath = document.getElementById(gaugeStrokeId);
        const scoreText = document.getElementById(scoreTextId);
        const badge = document.getElementById(badgeId);
        
        if (!strokePath || !scoreText || !badge) return;
        
        // Circular perimeter has length 100 for path coordinates.
        strokePath.setAttribute('stroke-dasharray', `${percent}, 100`);
        scoreText.innerText = `${percent}%`;
        
        badge.innerText = grade;
        badge.className = 'med-badge';
        
        const gr = grade.toLowerCase();
        if (gr.includes('high') || gr.includes('danger')) {
            strokePath.setAttribute('stroke', '#ef4444'); // red
            badge.classList.add('med-badge-danger');
        } else if (gr.includes('moderate') || gr.includes('intermediate') || gr.includes('borderline')) {
            strokePath.setAttribute('stroke', '#f59e0b'); // orange
            badge.classList.add('med-badge-warning');
        } else {
            strokePath.setAttribute('stroke', '#10b981'); // green
            badge.classList.add('med-badge-success');
        }
    }

    // Setup Telehealth room event listeners
    const telehealthCitySelect = document.getElementById('telehealthCitySelect');
    if (telehealthCitySelect) {
        telehealthCitySelect.addEventListener('change', () => {
            fetchNearbySpecialists();
        });
    }

    const requestConsultBtn = document.getElementById('requestConsultBtn');
    if (requestConsultBtn) {
        requestConsultBtn.addEventListener('click', () => {
            if (!currentScanId) return;
            
            const city = telehealthCitySelect?.value || 'Bangalore';
            const formData = new FormData();
            formData.append('city', city);
            
            const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]')?.value || '';
            formData.append('csrfmiddlewaretoken', csrfToken);
            
            requestConsultBtn.disabled = true;
            requestConsultBtn.innerHTML = `<span class="spinner-sm" style="display: inline-block; width: 12px; height: 12px; border: 2px solid #fff; border-top: 2px solid transparent; border-radius: 50%; animation: spin 1s linear infinite; margin-right: 6px;"></span> Assigning Specialist...`;
            
            fetch(`/api/consult/request/${currentScanId}/`, {
                method: 'POST',
                body: formData,
                headers: {
                    'X-CSRFToken': csrfToken
                }
            })
            .then(response => response.json())
            .then(data => {
                requestConsultBtn.disabled = false;
                requestConsultBtn.innerHTML = `<i class="ti ti-users-group"></i> Request Telehealth Consult Room`;
                if (data.success) {
                    currentConsultId = data.consult_id;
                    setupTelehealthRoom(data.consult_id, data.assigned_doctor, data.status, [], '', '');
                    // Start polling
                    if (telehealthPollInterval) clearInterval(telehealthPollInterval);
                    telehealthPollInterval = setInterval(loadConsultationMessages, 4000);
                } else {
                    alert('Error creating consult session: ' + data.error);
                }
            })
            .catch(error => {
                requestConsultBtn.disabled = false;
                requestConsultBtn.innerHTML = `<i class="ti ti-users-group"></i> Request Telehealth Consult Room`;
                console.error('Request consult error:', error);
                alert('A network error occurred.');
            });
        });
    }

    const telehealthChatForm = document.getElementById('telehealthChatForm');
    const telehealthChatInput = document.getElementById('telehealthChatInput');
    if (telehealthChatForm) {
        telehealthChatForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const text = telehealthChatInput?.value || '';
            if (!text.trim() || !currentConsultId) return;
            
            // Clear input
            telehealthChatInput.value = '';
            
            // Add bubble immediately for responsive UI
            addTelehealthMessageBubble('Patient', false, text, new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}));
            
            const formData = new FormData();
            formData.append('message', text);
            formData.append('sender', 'Patient');
            formData.append('is_doctor', 'false');
            
            const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]')?.value || '';
            formData.append('csrfmiddlewaretoken', csrfToken);
            
            fetch(`/api/consult/${currentConsultId}/messages/`, {
                method: 'POST',
                body: formData,
                headers: {
                    'X-CSRFToken': csrfToken
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    loadConsultationMessages();
                } else {
                    console.error('Send consult message failed:', data.error);
                }
            })
            .catch(error => console.error('Send message error:', error));
        });
    }

    const inviteSpecialistBtn = document.getElementById('inviteSpecialistBtn');
    const inviteSpecialistSelect = document.getElementById('inviteSpecialistSelect');
    if (inviteSpecialistBtn && inviteSpecialistSelect) {
        inviteSpecialistBtn.addEventListener('click', () => {
            const spec = inviteSpecialistSelect.value;
            if (!spec || !currentConsultId) return;
            
            const formData = new FormData();
            formData.append('specialist', spec);
            
            const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]')?.value || '';
            formData.append('csrfmiddlewaretoken', csrfToken);
            
            inviteSpecialistBtn.disabled = true;
            
            fetch(`/api/consult/${currentConsultId}/invite/`, {
                method: 'POST',
                body: formData,
                headers: {
                    'X-CSRFToken': csrfToken
                }
            })
            .then(response => response.json())
            .then(data => {
                inviteSpecialistBtn.disabled = false;
                if (data.success) {
                    loadConsultationMessages();
                } else {
                    alert('Error inviting specialist: ' + data.error);
                }
            })
            .catch(error => {
                inviteSpecialistBtn.disabled = false;
                console.error('Invite specialist error:', error);
            });
        });
    }

    const clinicalSignoffBtn = document.getElementById('clinicalSignoffBtn');
    const clinicalNotesInput = document.getElementById('clinicalNotesInput');
    const doctorSignatureInput = document.getElementById('doctorSignatureInput');
    if (clinicalSignoffBtn) {
        clinicalSignoffBtn.addEventListener('click', () => {
            const notes = clinicalNotesInput?.value || '';
            const sig = doctorSignatureInput?.value || '';
            
            if (!notes.trim() || !sig.trim()) {
                alert('Please enter both clinical sign-off notes and doctor signature.');
                return;
            }
            
            if (!currentConsultId) return;
            
            const formData = new FormData();
            formData.append('clinical_notes', notes);
            formData.append('doctor_signature', sig);
            
            const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]')?.value || '';
            formData.append('csrfmiddlewaretoken', csrfToken);
            
            clinicalSignoffBtn.disabled = true;
            
            fetch(`/api/consult/${currentConsultId}/signoff/`, {
                method: 'POST',
                body: formData,
                headers: {
                    'X-CSRFToken': csrfToken
                }
            })
            .then(response => response.json())
            .then(data => {
                clinicalSignoffBtn.disabled = false;
                if (data.success) {
                    loadConsultationMessages();
                } else {
                    alert('Sign-off error: ' + data.error);
                }
            })
            .catch(error => {
                clinicalSignoffBtn.disabled = false;
                console.error('Signoff error:', error);
            });
        });
    }

    function fetchNearbySpecialists() {
        if (!currentScanId) return;
        const city = telehealthCitySelect?.value || 'Bangalore';
        const formData = new FormData();
        formData.append('city', city);
        
        const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]')?.value || '';
        
        fetch(`/api/consult/request/${currentScanId}/`, {
            method: 'POST',
            body: formData,
            headers: {
                'X-CSRFToken': csrfToken
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateTelehealthReferrals(data.specialists);
            }
        })
        .catch(error => console.error('Fetch specialists error:', error));
    }

    function updateTelehealthReferrals(specialists) {
        const container = document.getElementById('specialistListContainer');
        if (!container) return;
        
        container.innerHTML = '';
        if (!specialists || specialists.length === 0) {
            container.innerHTML = `<p style="font-size: 11px; color: var(--text-secondary); grid-column: 1 / -1;">No specialists available in this area.</p>`;
            return;
        }
        
        specialists.forEach(d => {
            const card = document.createElement('div');
            card.style.cssText = `background: #fff; padding: 10px; border-radius: 6px; border: 0.5px solid #e5e7eb; display: flex; flex-direction: column; gap: 4px; transition: transform 0.2s, box-shadow 0.2s; cursor: pointer;`;
            card.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <span style="font-size: 11.5px; font-weight: 700; color: var(--text-primary);">${d.name}</span>
                    <span class="med-badge med-badge-success" style="font-size: 9px; padding: 1px 4px;">★ ${d.rating}</span>
                </div>
                <span style="font-size: 10px; color: var(--accent-primary); font-weight: 600;">${d.specialty} Specialist</span>
                <span style="font-size: 10px; color: var(--text-secondary); line-height: 1.3;">${d.hospital} (${d.city})</span>
                <div style="margin-top: auto; display: flex; justify-content: space-between; align-items: center; border-top: 0.5px solid #f3f4f6; padding-top: 4px; font-size: 9.5px; color: var(--text-secondary);">
                    <span><i class="ti ti-map-pin"></i> Proximity:</span>
                    <span style="font-weight: 600; color: var(--text-primary);">${d.distance}</span>
                </div>
            `;
            container.appendChild(card);
        });
    }

    function setupTelehealthRoom(consultId, principalDoc, status, panel, notes, signoff) {
        document.getElementById('telehealthReferralView').classList.add('hidden');
        document.getElementById('telehealthActiveRoom').classList.remove('hidden');
        
        document.getElementById('consultPrincipalDoc').innerText = principalDoc;
        const statusBadge = document.getElementById('consultStatusBadge');
        statusBadge.innerText = status;
        statusBadge.className = 'med-badge';
        
        if (status === 'COMPLETED') {
            statusBadge.classList.add('med-badge-success');
            document.getElementById('inviteSpecialistArea').classList.add('hidden');
            document.getElementById('clinicalSignoffBox').classList.add('hidden');
            document.getElementById('telehealthChatForm').classList.add('hidden');
        } else {
            statusBadge.classList.add('med-badge-warning');
            document.getElementById('inviteSpecialistArea').classList.remove('hidden');
            document.getElementById('clinicalSignoffBox').classList.remove('hidden');
            document.getElementById('telehealthChatForm').classList.remove('hidden');
        }
        
        // Populate Panel List
        const list = document.getElementById('consultPanelList');
        list.innerHTML = `<li>${principalDoc} (Principal Doctor)</li>`;
        (panel || []).forEach(name => {
            const li = document.createElement('li');
            li.innerText = name;
            list.appendChild(li);
        });

        // Setup Practo telemedicine CTA
        const practoContainer = document.getElementById('practoLinkContainer');
        const practoBtn = document.getElementById('practoLinkBtn');
        if (practoContainer && practoBtn) {
            practoContainer.classList.remove('hidden');
            let querySpecialist = "doctor";
            if (principalDoc.toLowerCase().includes('neurology') || principalDoc.toLowerCase().includes('sharma') || principalDoc.toLowerCase().includes('patel')) {
                querySpecialist = "neurologist";
            } else if (principalDoc.toLowerCase().includes('pulmonology') || principalDoc.toLowerCase().includes('mehta') || principalDoc.toLowerCase().includes('verma')) {
                querySpecialist = "pulmonologist";
            } else if (principalDoc.toLowerCase().includes('cardiology') || principalDoc.toLowerCase().includes('sen') || principalDoc.toLowerCase().includes('reddy')) {
                querySpecialist = "cardiologist";
            } else if (principalDoc.toLowerCase().includes('general') || principalDoc.toLowerCase().includes('rao') || principalDoc.toLowerCase().includes('gupta')) {
                querySpecialist = "general-physician";
            }
            practoBtn.href = `https://www.practo.com/search/doctors?results_type=doctor&q=${querySpecialist}&city=Bangalore`;
        }
    }

    function loadConsultationMessages() {
        if (!currentConsultId) return;
        
        fetch(`/api/consult/${currentConsultId}/messages/`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Populate room state
                setupTelehealthRoom(
                    currentConsultId,
                    data.assigned_doctor,
                    data.status,
                    data.specialist_panel,
                    data.clinical_notes,
                    data.signed_off_by
                );
                
                // Populate Chat messages
                const chatBox = document.getElementById('telehealthChatMessages');
                const atBottom = chatBox.scrollHeight - chatBox.clientHeight <= chatBox.scrollTop + 40;
                
                chatBox.innerHTML = '';
                (data.messages || []).forEach(m => {
                    addTelehealthMessageBubble(m.sender, m.is_doctor, m.message, m.timestamp);
                });
                
                // Auto scroll
                if (atBottom || chatBox.innerHTML === '') {
                    chatBox.scrollTop = chatBox.scrollHeight;
                }
                
                // Populate invite dropdown using specialists list
                const inviteSelect = document.getElementById('inviteSpecialistSelect');
                if (inviteSelect && inviteSelect.innerHTML === '') {
                    (data.specialists || []).forEach(d => {
                        const nameWithSpecialty = `${d.name} (${d.specialty})`;
                        if (nameWithSpecialty !== data.assigned_doctor.split(',')[0] && !data.specialist_panel.includes(nameWithSpecialty)) {
                            const opt = document.createElement('option');
                            opt.value = nameWithSpecialty;
                            opt.innerText = nameWithSpecialty;
                            inviteSelect.appendChild(opt);
                        }
                    });
                }
            }
        })
        .catch(error => console.error('Load messages error:', error));
    }

    function addTelehealthMessageBubble(sender, isDoctor, text, time) {
        const chatBox = document.getElementById('telehealthChatMessages');
        if (!chatBox) return;
        
        const div = document.createElement('div');
        div.style.cssText = `max-width: 80%; display: flex; flex-direction: column; gap: 2px; padding: 8px 12px; border-radius: 8px; font-size: 11.5px; line-height: 1.4;`;
        
        if (sender === 'System') {
            div.style.cssText += `align-self: center; background: #eef2f6; border: 0.5px solid #d0d7de; color: var(--text-secondary); text-align: center; font-size: 10px; width: 90%; max-width: 90%;`;
            div.innerHTML = `<strong>System Notice:</strong> ${text}`;
        } else if (isDoctor) {
            div.style.cssText += `align-self: flex-start; background: #fff; border: 0.5px solid #e5e7eb; color: var(--text-primary);`;
            div.innerHTML = `
                <strong style="font-size: 9px; color: var(--accent-primary); text-transform: uppercase; margin-bottom: 2px;">${sender}</strong>
                <span>${text}</span>
                <span style="font-size: 8.5px; color: var(--text-secondary); align-self: flex-end; margin-top: 2px;">${time}</span>
            `;
        } else {
            div.style.cssText += `align-self: flex-end; background: var(--accent-primary); color: #fff;`;
            div.innerHTML = `
                <span>${text}</span>
                <span style="font-size: 8.5px; color: rgba(255,255,255,0.7); align-self: flex-end; margin-top: 2px;">${time}</span>
            `;
        }
        
        chatBox.appendChild(div);
    }

    // Setup Vitals Synchronizer & Progress Timeline listeners
    const syncVitalsBtn = document.getElementById('syncVitalsBtn');
    if (syncVitalsBtn) {
        syncVitalsBtn.addEventListener('click', () => {
            syncVitalsBtn.disabled = true;
            syncVitalsBtn.innerHTML = `<span class="spinner-sm" style="display: inline-block; width: 10px; height: 10px; border: 2px solid #fff; border-top: 2px solid transparent; border-radius: 50%; animation: spin 1s linear infinite; margin-right: 4px;"></span> Syncing...`;
            
            setTimeout(() => {
                const hr = Math.floor(Math.random() * (85 - 65 + 1)) + 65;
                const steps = Math.floor(Math.random() * (12000 - 6000 + 1)) + 6000;
                const sleep = (Math.random() * (8.5 - 6.0) + 6.0).toFixed(1);
                
                document.getElementById('vitalsHrText').innerText = `${hr} bpm`;
                document.getElementById('vitalsStepsText').innerText = `${steps.toLocaleString()} steps`;
                document.getElementById('vitalsSleepText').innerText = `${sleep} hrs`;
                
                syncVitalsBtn.disabled = false;
                syncVitalsBtn.innerHTML = `<i class="ti ti-refresh"></i> Sync Vitals`;
            }, 1000);
        });
    }

    function updatePatientHistoryTimeline() {
        fetch('/api/patient/history/')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const container = document.getElementById('progressTimelineContainer');
                if (!container) return;
                
                container.innerHTML = '';
                if (!data.history || data.history.length === 0) {
                    container.innerHTML = `<p style="font-size: 11px; color: var(--text-secondary); margin: 0;">Upload multiple scans to track indicators over time.</p>`;
                    return;
                }
                
                data.history.forEach((scan, index) => {
                    const row = document.createElement('div');
                    row.style.cssText = `display: flex; gap: 10px; align-items: center; background: #fff; padding: 8px 10px; border-radius: 6px; border: 0.5px solid #e5e7eb; font-size: 11.5px; position: relative; margin-top: 4px;`;
                    
                    const indicatorDot = document.createElement('div');
                    indicatorDot.style.cssText = `width: 8px; height: 8px; border-radius: 50%; background: ${scan.detected ? '#ef4444' : '#10b981'}; flex-shrink: 0;`;
                    
                    const textBlock = document.createElement('div');
                    textBlock.style.cssText = `flex: 1; display: flex; justify-content: space-between; align-items: center;`;
                    
                    const labelSpan = document.createElement('span');
                    labelSpan.innerHTML = `<strong>Scan #${index + 1}:</strong> ${scan.modality} (${scan.filename}) <span style="font-size: 9.5px; color: var(--text-secondary); margin-left: 4px;">${scan.timestamp}</span>`;
                    
                    const valSpan = document.createElement('span');
                    valSpan.style.cssText = `font-weight: 700; color: var(--text-primary);`;
                    valSpan.innerText = `Metric: ${scan.confidence}%`;
                    
                    textBlock.appendChild(labelSpan);
                    textBlock.appendChild(valSpan);
                    
                    row.appendChild(indicatorDot);
                    row.appendChild(textBlock);
                    
                    container.appendChild(row);
                });
            }
        })
        .catch(error => console.error('Patient history timeline error:', error));
    }
});
