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

    resetBtn.addEventListener('click', () => {
        resultsArea.classList.add('hidden');
        uploadArea.classList.remove('hidden');
        fileInput.value = '';
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

        const formData = new FormData();
        formData.append('image', file);

        fetch(predictApiUrl, {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            loadingState.classList.add('hidden');
            
            if (data.success) {
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
});
