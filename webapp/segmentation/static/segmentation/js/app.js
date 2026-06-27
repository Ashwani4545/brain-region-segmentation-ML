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
        });
    });

    // Reset button handler
    resetBtn.addEventListener('click', () => {
        resultsArea.classList.add('hidden');
        uploadArea.classList.remove('hidden');
        fileInput.value = '';
        currentScanId = null;
        document.getElementById('chatMessages').innerHTML = '';
        
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
});
