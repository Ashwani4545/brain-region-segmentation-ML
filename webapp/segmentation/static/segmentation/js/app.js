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

    // Reset button handler
    resetBtn.addEventListener('click', () => {
        resultsArea.classList.add('hidden');
        uploadArea.classList.remove('hidden');
        fileInput.value = '';
        currentScanId = null;
        document.getElementById('chatMessages').innerHTML = '';
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
                
                // Populate images
                document.getElementById('origImage').src = data.original_url;
                document.getElementById('maskImage').src = data.mask_url;
                document.getElementById('overlayImage').src = data.overlay_url;
                
                // Populate confidence badge
                const badge = document.getElementById('confidenceBadge');
                badge.innerText = `Confidence: ${data.confidence}`;
                
                if (data.detected) {
                    badge.style.color = '#ef4444'; // Red showing danger/finding
                    badge.style.borderColor = 'rgba(239, 68, 68, 0.2)';
                    badge.style.background = 'rgba(239, 68, 68, 0.1)';
                    document.getElementById('explainText').innerText = `Anomaly Detected. The system detected hypodense regions indicating possible stroke-affected areas. Confidence level: ${data.confidence}. See overlay for exact visual localization.`;
                } else {
                    badge.style.color = '#10b981'; // Green showing clean
                    badge.style.borderColor = 'rgba(16, 185, 129, 0.2)';
                    badge.style.background = 'rgba(16, 185, 129, 0.1)';
                    document.getElementById('explainText').innerText = `No Analysis Findings. The model did not detect significant hypodense regions in this scan slice.`;
                }
                
                // Update download link
                document.getElementById('downloadBtn').href = data.overlay_url;

                resultsArea.classList.remove('hidden');

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
        const msg = data.detected 
            ? `Hello, I am MedAssist, your compassionate care companion. I see that the AI analysis flagged some hypodense regions (confidence: ${data.confidence}). I understand this can be concerning or cause anxiety. Please know I am here to help explain what these terms mean and guide you. How are you feeling right now?`
            : `Hello, I am MedAssist, your compassionate care companion. The scan has been processed and did not show significant hypodensities (confidence: ${data.confidence}). How can I help you today?`;
            
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
});
