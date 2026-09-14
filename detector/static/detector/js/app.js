/**
 * Facial Expression Recognition Studio - Frontend Controller
 * Integrates Live Webcam, Model Inference APIs, Chart.js Visualizers & History
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- State Management ---
    const state = {
        activeTab: 'live-studio',
        isStreaming: false,
        autoDetect: true,
        detectIntervalMs: 500,
        selectedModel: 'face_emotion_model.onnx',
        lastEmotion: null,
        soundEnabled: false,
        fps: 0,
        frameCount: 0,
        lastFpsTime: performance.now(),
        radarChart: null,
        historyFilter: { emotion: 'all', source: 'all' },
        currentUploadFaces: [],
        selectedFaceIdx: 0
    };

    // 8 Emotion definitions & metadata
    const EMOTION_META = {
        'Happy': { emoji: '😃', color: '#10b981', glow: 'rgba(16,185,129,0.4)', quote: '“Keep smiling, because life is a beautiful thing and there’s so much to smile about.”', vibe: 'Upbeat Indie Pop / Sunshine Funk' },
        'Surprise': { emoji: '😲', color: '#f59e0b', glow: 'rgba(245,158,11,0.4)', quote: '“The mind loves the unknown. It loves images whose meaning is unknown.”', vibe: 'Electronic Future Bass / High Energy' },
        'Suprise': { emoji: '😲', color: '#f59e0b', glow: 'rgba(245,158,11,0.4)', quote: '“The mind loves the unknown. It loves images whose meaning is unknown.”', vibe: 'Electronic Future Bass / High Energy' },
        'Sad': { emoji: '😢', color: '#06b6d4', glow: 'rgba(6,182,212,0.4)', quote: '“Tears are words that need to be written. Take a gentle breath.”', vibe: 'Acoustic Lo-Fi / Ambient Chill' },
        'Angry': { emoji: '😠', color: '#f43f5e', glow: 'rgba(244,63,94,0.4)', quote: '“Speak when you are angry and you will make the best speech you will ever regret.”', vibe: 'Hard Rock / Intense Synthwave' },
        'Fear': { emoji: '😨', color: '#8b5cf6', glow: 'rgba(139,92,246,0.4)', quote: '“Courage is resistance to fear, mastery of fear, not absence of fear.”', vibe: 'Calming Deep Meditative Drone' },
        'Contempt': { emoji: '😒', color: '#ec4899', glow: 'rgba(236,72,153,0.4)', quote: '“The highest form of knowledge is empathy. Stay open and grounded.”', vibe: 'Alternative R&B / Deep Chill' },
        'Disgust': { emoji: '🤢', color: '#84cc16', glow: 'rgba(132,204,22,0.4)', quote: '“Turn aversion into understanding. Reset your mindset.”', vibe: 'Nature Sounds / Pure Forest Rain' },
        'Neutral': { emoji: '😐', color: '#94a3b8', glow: 'rgba(148,163,184,0.4)', quote: '“Calmness is the cradle of power.”', vibe: 'Smooth Jazz / Coffee House Beats' }
    };

    // --- DOM Elements ---
    const videoElem = document.getElementById('webcam-video');
    const overlayCanvas = document.getElementById('overlay-canvas');
    const overlayCtx = overlayCanvas ? overlayCanvas.getContext('2d') : null;
    const placeholderElem = document.getElementById('camera-placeholder');
    const btnToggleCam = document.getElementById('btn-toggle-cam');
    const btnCaptureSnap = document.getElementById('btn-capture-snap');
    const autoDetectSwitch = document.getElementById('switch-auto-detect');
    const soundSwitch = document.getElementById('switch-sound');
    const modelSelector = document.getElementById('global-model-select');
    const fpsBadge = document.getElementById('fps-badge');
    const latencyBadge = document.getElementById('latency-badge');

    // Hero elements
    const heroAvatar = document.getElementById('hero-avatar');
    const heroEmotionName = document.getElementById('hero-emotion-name');
    const heroConfidence = document.getElementById('hero-confidence');
    const heroQuote = document.getElementById('hero-quote');
    const moodVibeText = document.getElementById('mood-vibe-text');

    // --- Sound Synthesis (Web Audio API) ---
    let audioCtx = null;
    function playChime(emotion) {
        if (!state.soundEnabled) return;
        try {
            if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            osc.connect(gain);
            gain.connect(audioCtx.destination);

            const freqs = { 'Happy': 523.25, 'Suprise': 659.25, 'Sad': 329.63, 'Angry': 220.00, 'Fear': 440.00 };
            osc.frequency.setValueAtTime(freqs[emotion] || 440, audioCtx.currentTime);
            gain.gain.setValueAtTime(0.08, audioCtx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.35);

            osc.start();
            osc.stop(audioCtx.currentTime + 0.35);
        } catch (e) {
            console.warn("Audio chime notice:", e);
        }
    }

    // --- Toast Notification ---
    function showToast(message, icon = '✨') {
        let container = document.querySelector('.toast-container');
        if (!container) {
            container = document.createElement('div');
            container.className = 'toast-container';
            document.body.appendChild(container);
        }
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.innerHTML = `<span>${icon}</span> <span>${message}</span>`;
        container.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(10px)';
            setTimeout(() => toast.remove(), 300);
        }, 3200);
    }

    // --- Tab Navigation ---
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanels = document.querySelectorAll('.tab-panel');

    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.dataset.tab;
            state.activeTab = targetTab;

            tabButtons.forEach(b => b.classList.toggle('active', b === btn));
            tabPanels.forEach(p => p.classList.toggle('active', p.id === `tab-${targetTab}`));

            if (targetTab === 'analytics') {
                updateRadarChart();
            } else if (targetTab === 'history') {
                loadHistory();
            }
        });
    });

    // --- Drawer & Mobile Menu Controls ---
    const btnMobileMenu = document.getElementById('btn-mobile-menu');
    const btnCloseDrawer = document.getElementById('btn-close-drawer');
    const mobileDrawer = document.getElementById('mobile-menu-drawer');
    const drawerBackdrop = document.getElementById('drawer-backdrop');

    function openDrawer() {
        if (mobileDrawer) mobileDrawer.classList.add('open');
        if (drawerBackdrop) drawerBackdrop.classList.add('open');
    }

    function closeDrawer() {
        if (mobileDrawer) mobileDrawer.classList.remove('open');
        if (drawerBackdrop) drawerBackdrop.classList.remove('open');
    }

    if (btnMobileMenu) btnMobileMenu.addEventListener('click', openDrawer);
    if (btnCloseDrawer) btnCloseDrawer.addEventListener('click', closeDrawer);
    if (drawerBackdrop) drawerBackdrop.addEventListener('click', closeDrawer);

    // --- Model Selector Two-Way Sync (Desktop & Mobile Drawer) ---
    const modelDropdowns = document.querySelectorAll('.model-dropdown-sync');
    modelDropdowns.forEach(dropdown => {
        dropdown.addEventListener('change', (e) => {
            const selectedVal = e.target.value;
            state.selectedModel = selectedVal;

            // Sync other dropdowns
            modelDropdowns.forEach(other => {
                if (other !== e.target) {
                    other.value = selectedVal;
                }
            });

            const modelName = e.target.selectedOptions[0] ? e.target.selectedOptions[0].text : selectedVal;
            showToast(`Active Model: ${modelName}`, '🧠');
        });
    });

    // --- Switch Controls Two-Way Sync ---
    const mobileAutoDetectSwitch = document.getElementById('mobile-switch-auto-detect');
    const mobileSoundSwitch = document.getElementById('mobile-switch-sound');

    if (mobileAutoDetectSwitch && autoDetectSwitch) {
        mobileAutoDetectSwitch.addEventListener('change', (e) => {
            autoDetectSwitch.checked = e.target.checked;
            autoDetectSwitch.dispatchEvent(new Event('change'));
        });
        autoDetectSwitch.addEventListener('change', (e) => {
            mobileAutoDetectSwitch.checked = e.target.checked;
        });
    }

    if (mobileSoundSwitch && soundSwitch) {
        mobileSoundSwitch.addEventListener('change', (e) => {
            soundSwitch.checked = e.target.checked;
            soundSwitch.dispatchEvent(new Event('change'));
        });
        soundSwitch.addEventListener('change', (e) => {
            mobileSoundSwitch.checked = e.target.checked;
        });
    }

    // --- Webcam Streaming ---
    let streamMedia = null;
    let detectionTimer = null;
    const btnFlipCam = document.getElementById('btn-flip-cam');

    async function startCamera() {
        try {
            if (streamMedia) {
                streamMedia.getTracks().forEach(track => track.stop());
            }

            const constraints = {
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: state.facingMode || 'user'
                },
                audio: false
            };

            streamMedia = await navigator.mediaDevices.getUserMedia(constraints);
            videoElem.srcObject = streamMedia;
            videoElem.style.display = 'block';
            if (placeholderElem) placeholderElem.style.display = 'none';
            if (overlayCanvas) overlayCanvas.style.display = 'block';

            // Flip mirror if rear camera
            const isFront = (state.facingMode || 'user') === 'user';
            videoElem.style.transform = isFront ? 'scaleX(-1)' : 'none';
            if (overlayCanvas) overlayCanvas.style.transform = isFront ? 'scaleX(-1)' : 'none';

            state.isStreaming = true;
            btnToggleCam.innerHTML = `<span class="btn-icon">⏹️</span>`;
            btnToggleCam.title = "Stop Camera";
            btnToggleCam.setAttribute('aria-label', 'Stop Camera');
            btnToggleCam.classList.remove('btn-primary');
            btnToggleCam.classList.add('btn-danger');

            if (btnFlipCam) btnFlipCam.style.display = 'inline-flex';

            videoElem.onloadedmetadata = () => {
                videoElem.play();
                if (overlayCanvas) {
                    overlayCanvas.width = videoElem.videoWidth || 640;
                    overlayCanvas.height = videoElem.videoHeight || 480;
                }
                startDetectionLoop();
            };
            showToast('Camera started successfully!', '📹');
        } catch (err) {
            console.error("Camera access error:", err);
            showToast('Unable to access camera: ' + err.message, '⚠️');
        }
    }

    function stopCamera() {
        if (streamMedia) {
            streamMedia.getTracks().forEach(track => track.stop());
            streamMedia = null;
        }
        if (detectionTimer) {
            clearTimeout(detectionTimer);
            detectionTimer = null;
        }
        videoElem.style.display = 'none';
        if (placeholderElem) placeholderElem.style.display = 'flex';
        if (overlayCanvas) {
            overlayCanvas.style.display = 'none';
            overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        }

        state.isStreaming = false;
        btnToggleCam.innerHTML = `<span class="btn-icon">▶️</span>`;
        btnToggleCam.title = "Start Camera";
        btnToggleCam.setAttribute('aria-label', 'Start Camera');
        btnToggleCam.classList.remove('btn-danger');
        btnToggleCam.classList.add('btn-primary');
        if (btnFlipCam) btnFlipCam.style.display = 'none';
        fpsBadge.textContent = '0 FPS';
    }

    if (btnToggleCam) {
        btnToggleCam.addEventListener('click', () => {
            if (state.isStreaming) stopCamera();
            else startCamera();
        });
    }

    if (btnFlipCam) {
        btnFlipCam.addEventListener('click', async () => {
            if (!state.isStreaming) return;
            state.facingMode = (state.facingMode === 'environment') ? 'user' : 'environment';
            showToast(`Switching to ${state.facingMode === 'user' ? 'Front' : 'Rear'} camera...`, '🔄');
            await startCamera();
        });
    }

    if (autoDetectSwitch) {
        autoDetectSwitch.addEventListener('change', (e) => {
            state.autoDetect = e.target.checked;
            if (state.autoDetect && state.isStreaming && !detectionTimer) {
                startDetectionLoop();
            }
        });
    }

    if (soundSwitch) {
        soundSwitch.addEventListener('change', (e) => {
            state.soundEnabled = e.target.checked;
            if (state.soundEnabled) playChime('Happy');
        });
    }

    // --- Frame Capture & Inference Loop ---
    const captureCanvas = document.createElement('canvas');
    const captureCtx = captureCanvas.getContext('2d');

    function getFrameBase64(scale = 0.8) {
        if (!videoElem || !videoElem.videoWidth) return null;
        captureCanvas.width = videoElem.videoWidth * scale;
        captureCanvas.height = videoElem.videoHeight * scale;
        captureCtx.drawImage(videoElem, 0, 0, captureCanvas.width, captureCanvas.height);
        return captureCanvas.toDataURL('image/jpeg', 0.85);
    }

    let isPredicting = false;

    async function detectLiveFrame(saveRecord = false) {
        if (!state.isStreaming || isPredicting) return;
        const b64Data = getFrameBase64();
        if (!b64Data) return;

        isPredicting = true;
        const t0 = performance.now();

        try {
            const formData = new FormData();
            formData.append('image_base64', b64Data);
            formData.append('model_name', state.selectedModel);
            formData.append('source', saveRecord ? 'webcam_snap' : 'webcam_live');
            formData.append('save_record', saveRecord ? 'true' : 'false');
            formData.append('draw_annotations', 'false');

            const res = await fetch('/api/predict/', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();

            if (data.success) {
                renderFaceBoxes(data.faces);
                updateEmotionUI(data.primary_emotion, data.confidence, data.probabilities);
                
                const tEnd = performance.now();
                latencyBadge.textContent = `${Math.round(tEnd - t0)} ms`;

                // Calculate FPS
                state.frameCount++;
                const now = performance.now();
                if (now - state.lastFpsTime >= 1000) {
                    state.fps = Math.round((state.frameCount * 1000) / (now - state.lastFpsTime));
                    fpsBadge.textContent = `${state.fps} FPS`;
                    state.frameCount = 0;
                    state.lastFpsTime = now;
                }

                if (data.primary_emotion !== state.lastEmotion) {
                    playChime(data.primary_emotion);
                    state.lastEmotion = data.primary_emotion;
                }
            }
        } catch (err) {
            console.error("Predict frame error:", err);
        } finally {
            isPredicting = false;
        }
    }

    function startDetectionLoop() {
        if (!state.isStreaming || !state.autoDetect) return;
        detectLiveFrame(false).finally(() => {
            if (state.isStreaming && state.autoDetect) {
                detectionTimer = setTimeout(startDetectionLoop, state.detectIntervalMs);
            }
        });
    }

    if (btnCaptureSnap) {
        btnCaptureSnap.addEventListener('click', async () => {
            if (!state.isStreaming) {
                showToast('Start camera before capturing snapshot', '⚠️');
                return;
            }
            btnCaptureSnap.disabled = true;
            btnCaptureSnap.innerHTML = `<span class="btn-icon">⏳</span>`;
            await detectLiveFrame(true);
            showToast('Snapshot saved to history!', '📸');
            btnCaptureSnap.disabled = false;
            btnCaptureSnap.innerHTML = `<span class="btn-icon">📸</span>`;
        });
    }

    // --- Canvas Overlay Rendering ---
    function renderFaceBoxes(faces) {
        if (!overlayCtx || !overlayCanvas) return;
        overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

        if (!faces || faces.length === 0) return;

        const scaleX = overlayCanvas.width / (videoElem.videoWidth * 0.8 || overlayCanvas.width);
        const scaleY = overlayCanvas.height / (videoElem.videoHeight * 0.8 || overlayCanvas.height);

        faces.forEach(face => {
            const [x, y, w, h] = face.box.map((v, i) => i % 2 === 0 ? v * scaleX : v * scaleY);
            const color = face.color || '#38bdf8';

            // Glow box
            overlayCtx.strokeStyle = color;
            overlayCtx.lineWidth = 2.5;
            overlayCtx.shadowColor = color;
            overlayCtx.shadowBlur = 12;
            overlayCtx.strokeRect(x, y, w, h);

            // Corner Brackets
            const cornerSize = Math.min(22, w / 4, h / 4);
            overlayCtx.strokeStyle = '#ffffff';
            overlayCtx.lineWidth = 3.5;
            overlayCtx.shadowBlur = 0;

            // Top-Left
            overlayCtx.beginPath();
            overlayCtx.moveTo(x, y + cornerSize);
            overlayCtx.lineTo(x, y);
            overlayCtx.lineTo(x + cornerSize, y);
            overlayCtx.stroke();

            // Top-Right
            overlayCtx.beginPath();
            overlayCtx.moveTo(x + w - cornerSize, y);
            overlayCtx.lineTo(x + w, y);
            overlayCtx.lineTo(x + w, y + cornerSize);
            overlayCtx.stroke();

            // Bottom-Left
            overlayCtx.beginPath();
            overlayCtx.moveTo(x, y + h - cornerSize);
            overlayCtx.lineTo(x, y + h);
            overlayCtx.lineTo(x + cornerSize, y + h);
            overlayCtx.stroke();

            // Bottom-Right
            overlayCtx.beginPath();
            overlayCtx.moveTo(x + w - cornerSize, y + h);
            overlayCtx.lineTo(x + w, y + h);
            overlayCtx.lineTo(x + w, y + h - cornerSize);
            overlayCtx.stroke();

            // Label pill
            const label = `${face.emoji || ''} ${face.primary_emotion} ${Math.round(face.confidence)}%`;
            overlayCtx.font = 'bold 13px Inter, sans-serif';
            const textWidth = overlayCtx.measureText(label).width;

            overlayCtx.fillStyle = color;
            overlayCtx.fillRect(x, Math.max(0, y - 24), textWidth + 16, 24);

            overlayCtx.fillStyle = '#ffffff';
            overlayCtx.fillText(label, x + 8, Math.max(16, y - 7));
        });
    }

    // --- Emotion UI & Probability Bars Update ---
    function updateEmotionUI(primaryEmotion, confidence, probabilities) {
        const meta = EMOTION_META[primaryEmotion] || EMOTION_META[primaryEmotion === 'Suprise' ? 'Surprise' : 'Neutral'] || EMOTION_META['Neutral'];

        // Update Live Studio Hero
        if (heroAvatar) {
            heroAvatar.textContent = meta.emoji;
            heroAvatar.style.borderColor = meta.color;
            heroAvatar.style.boxShadow = `0 0 25px ${meta.glow}`;
        }
        if (heroEmotionName) {
            heroEmotionName.textContent = primaryEmotion;
            heroEmotionName.style.color = meta.color;
        }
        if (heroConfidence) {
            heroConfidence.textContent = `${confidence}%`;
            heroConfidence.style.color = meta.color;
            heroConfidence.style.borderColor = meta.color;
        }
        if (heroQuote) {
            heroQuote.textContent = meta.quote;
        }
        if (moodVibeText) {
            moodVibeText.textContent = meta.vibe;
        }

        // Update Upload Tab Hero
        const uploadAvatar = document.getElementById('upload-hero-avatar');
        const uploadEmotionName = document.getElementById('upload-hero-emotion-name');
        const uploadConfidence = document.getElementById('upload-hero-confidence');
        const uploadQuote = document.getElementById('upload-hero-quote');

        if (uploadAvatar) {
            uploadAvatar.textContent = meta.emoji;
            uploadAvatar.style.borderColor = meta.color;
            uploadAvatar.style.boxShadow = `0 0 25px ${meta.glow}`;
        }
        if (uploadEmotionName) {
            uploadEmotionName.textContent = primaryEmotion;
            uploadEmotionName.style.color = meta.color;
        }
        if (uploadConfidence) {
            uploadConfidence.textContent = `${confidence}%`;
            uploadConfidence.style.color = meta.color;
            uploadConfidence.style.borderColor = meta.color;
        }
        if (uploadQuote) {
            uploadQuote.textContent = meta.quote;
        }

        // Update probability progress bars in BOTH tabs
        if (probabilities) {
            Object.entries(probabilities).forEach(([emotion, val]) => {
                const altKey = (emotion === 'Surprise') ? 'Suprise' : (emotion === 'Suprise') ? 'Surprise' : emotion;

                // Live studio bars
                const fillBar = document.getElementById(`prob-bar-${emotion}`) || document.getElementById(`prob-bar-${altKey}`);
                const valElem = document.getElementById(`prob-val-${emotion}`) || document.getElementById(`prob-val-${altKey}`);
                if (fillBar) fillBar.style.width = `${Math.min(100, Math.max(0, val))}%`;
                if (valElem) valElem.textContent = `${val}%`;

                // Upload tab bars
                const upFillBar = document.getElementById(`upload-prob-bar-${emotion}`) || document.getElementById(`upload-prob-bar-${altKey}`);
                const upValElem = document.getElementById(`upload-prob-val-${emotion}`) || document.getElementById(`upload-prob-val-${altKey}`);
                if (upFillBar) upFillBar.style.width = `${Math.min(100, Math.max(0, val))}%`;
                if (upValElem) upValElem.textContent = `${val}%`;
            });

            // Update Radar Chart if active
            if (state.radarChart) {
                const emotionLabels = ['Angry', 'Fear', 'Happy', 'Sad', 'Suprise'];
                state.radarChart.data.datasets[0].data = emotionLabels.map(k => probabilities[k] || probabilities[k === 'Suprise' ? 'Surprise' : 'Suprise'] || 0);
                state.radarChart.update('none');
            }
        }
    }

    // --- Chart.js Radar Initialization ---
    function initRadarChart() {
        const ctx = document.getElementById('emotion-radar-chart');
        if (!ctx) return;

        const emotionLabels = ['Angry', 'Fear', 'Happy', 'Sad', 'Suprise'];
        state.radarChart = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: emotionLabels,
                datasets: [{
                    label: 'Emotion Intensity (%)',
                    data: [15, 10, 50, 15, 10],
                    backgroundColor: 'rgba(56, 189, 248, 0.25)',
                    borderColor: '#38bdf8',
                    borderWidth: 2,
                    pointBackgroundColor: '#ffffff',
                    pointBorderColor: '#38bdf8',
                    pointHoverBackgroundColor: '#38bdf8',
                    pointHoverBorderColor: '#ffffff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                        grid: { color: 'rgba(255, 255, 255, 0.08)' },
                        pointLabels: {
                            color: '#94a3b8',
                            font: { size: 12, weight: 'bold' }
                        },
                        ticks: {
                            backdropColor: 'transparent',
                            color: '#64748b',
                            stepSize: 20
                        },
                        suggestedMin: 0,
                        suggestedMax: 100
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }

    function updateRadarChart() {
        if (!state.radarChart) initRadarChart();
    }

    initRadarChart();

    // --- Image File Upload Handling ---
    const dropzone = document.getElementById('upload-dropzone');
    const fileInput = document.getElementById('image-file-input');
    const uploadPreviewContainer = document.getElementById('upload-preview-container');
    const uploadPreviewImg = document.getElementById('uploaded-preview-img');
    const multiFaceGallery = document.getElementById('multi-face-gallery');

    if (dropzone && fileInput) {
        dropzone.addEventListener('click', () => fileInput.click());
        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        });
        dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                processUploadedFile(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files && fileInput.files.length > 0) {
                processUploadedFile(fileInput.files[0]);
            }
        });
    }

    async function processUploadedFile(file) {
        if (!file.type.startsWith('image/')) {
            showToast('Please upload an image file (JPG, PNG, WebP)', '⚠️');
            return;
        }

        showToast(`Processing ${file.name}...`, '⏳');
        const formData = new FormData();
        formData.append('image', file);
        formData.append('model_name', state.selectedModel);
        formData.append('source', 'image_upload');
        formData.append('save_record', 'true');
        formData.append('draw_annotations', 'true');

        try {
            const res = await fetch('/api/predict/', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();

            if (data.success) {
                state.currentUploadFaces = data.faces;
                
                // Show annotated image in preview
                if (uploadPreviewImg && data.annotated_image) {
                    uploadPreviewImg.src = data.annotated_image;
                    uploadPreviewImg.classList.add('no-mirror');
                    if (dropzone) dropzone.style.display = 'none';
                    if (uploadPreviewContainer) uploadPreviewContainer.style.display = 'block';
                }

                // Render multi-face gallery
                renderUploadFaceGallery(data.faces);

                // Update primary emotion
                updateEmotionUI(data.primary_emotion, data.confidence, data.probabilities);
                showToast(`Detected ${data.face_count} face(s) with primary mood: ${data.primary_emotion}`, '🎉');
            } else {
                showToast(`Detection error: ${data.error}`, '❌');
            }
        } catch (err) {
            console.error("Upload error:", err);
            showToast('Failed to process image: ' + err.message, '❌');
        }
    }

    function renderUploadFaceGallery(faces) {
        if (!multiFaceGallery) return;
        multiFaceGallery.innerHTML = '';

        if (!faces || faces.length <= 1) {
            multiFaceGallery.style.display = 'none';
            return;
        }

        multiFaceGallery.style.display = 'grid';
        faces.forEach((face, idx) => {
            const card = document.createElement('div');
            card.className = `face-thumb-card ${idx === 0 ? 'selected' : ''}`;
            card.innerHTML = `
                <img src="${face.face_crop_base64}" class="face-thumb-img" alt="Face ${face.face_id}">
                <div class="face-thumb-emotion" style="color: ${face.color};">${face.emoji} ${face.primary_emotion}</div>
                <div class="face-thumb-conf">${face.confidence}%</div>
            `;
            card.addEventListener('click', () => {
                document.querySelectorAll('.face-thumb-card').forEach(c => c.classList.remove('selected'));
                card.classList.add('selected');
                updateEmotionUI(face.primary_emotion, face.confidence, face.probabilities);
            });
            multiFaceGallery.appendChild(card);
        });
    }

    // Reset upload button
    const btnResetUpload = document.getElementById('btn-reset-upload');
    if (btnResetUpload) {
        btnResetUpload.addEventListener('click', () => {
            if (fileInput) fileInput.value = '';
            if (uploadPreviewContainer) uploadPreviewContainer.style.display = 'none';
            if (dropzone) dropzone.style.display = 'flex';
            if (multiFaceGallery) multiFaceGallery.style.display = 'none';

            // Reset upload hero and bars
            const uploadAvatar = document.getElementById('upload-hero-avatar');
            const uploadEmotionName = document.getElementById('upload-hero-emotion-name');
            const uploadConfidence = document.getElementById('upload-hero-confidence');
            const uploadQuote = document.getElementById('upload-hero-quote');

            if (uploadAvatar) {
                uploadAvatar.textContent = '🎭';
                uploadAvatar.style.borderColor = 'var(--primary-accent)';
                uploadAvatar.style.boxShadow = 'none';
            }
            if (uploadEmotionName) {
                uploadEmotionName.textContent = 'Awaiting Image';
                uploadEmotionName.style.color = 'var(--text-primary)';
            }
            if (uploadConfidence) {
                uploadConfidence.textContent = '0.0%';
                uploadConfidence.style.color = 'var(--primary-accent)';
                uploadConfidence.style.borderColor = 'rgba(56, 189, 248, 0.3)';
            }
            if (uploadQuote) {
                uploadQuote.textContent = 'Upload a photo above to run face detection and classify facial expressions.';
            }

            const classes = ['Angry', 'Fear', 'Happy', 'Sad', 'Suprise', 'Surprise'];
            classes.forEach(c => {
                const bar = document.getElementById(`upload-prob-bar-${c}`);
                const val = document.getElementById(`upload-prob-val-${c}`);
                if (bar) bar.style.width = '0%';
                if (val) val.textContent = '0.0%';
            });
        });
    }

    // --- Detection History Management ---
    const historyTableBody = document.getElementById('history-table-body');
    const filterEmotion = document.getElementById('filter-emotion');
    const filterSource = document.getElementById('filter-source');
    const btnClearHistory = document.getElementById('btn-clear-history');
    const btnExportCsv = document.getElementById('btn-export-csv');
    const btnExportJson = document.getElementById('btn-export-json');

    async function loadHistory() {
        if (!historyTableBody) return;
        historyTableBody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted);">Loading logs...</td></tr>`;

        const emotion = filterEmotion ? filterEmotion.value : 'all';
        const source = filterSource ? filterSource.value : 'all';

        try {
            const res = await fetch(`/api/history/?emotion=${emotion}&source=${source}&limit=50`);
            const data = await res.json();

            if (data.success && data.records.length > 0) {
                historyTableBody.innerHTML = data.records.map(r => {
                    const meta = EMOTION_META[r.primary_emotion] || EMOTION_META['Neutral'];
                    const thumb = r.face_crop_url ? `<img src="${r.face_crop_url}" class="table-thumb" alt="Face">` : `<div class="table-thumb" style="display:flex;align-items:center;justify-content:center;font-size:20px;">${meta.emoji}</div>`;
                    const srcLabel = r.source === 'webcam_live' ? 'Live Stream' : r.source === 'webcam_snap' ? 'Webcam Snap' : 'Photo Upload';

                    return `
                        <tr>
                            <td>${thumb}</td>
                            <td><strong>${r.created_at}</strong></td>
                            <td><span class="badge" style="background: rgba(255,255,255,0.06);">${srcLabel}</span></td>
                            <td>
                                <span class="badge" style="background: ${meta.color}22; color: ${meta.color}; border: 1px solid ${meta.color}55;">
                                    ${meta.emoji} ${r.primary_emotion}
                                </span>
                            </td>
                            <td><strong>${r.confidence}%</strong></td>
                            <td><span style="color: var(--text-muted); font-size: 11px;">${r.model_name}</span></td>
                            <td>
                                <button class="btn btn-danger btn-sm btn-icon-only btn-delete-row" data-id="${r.id}" title="Delete record">
                                    🗑️
                                </button>
                            </td>
                        </tr>
                    `;
                }).join('');

                // Bind delete buttons
                document.querySelectorAll('.btn-delete-row').forEach(btn => {
                    btn.addEventListener('click', async (e) => {
                        const id = btn.dataset.id;
                        await fetch(`/api/history/delete/${id}/`, { method: 'POST' });
                        showToast(`Record #${id} deleted`, '🗑️');
                        loadHistory();
                    });
                });
            } else {
                historyTableBody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted); padding: 30px;">No detection records found.</td></tr>`;
            }
        } catch (err) {
            console.error("History fetch error:", err);
            historyTableBody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--color-angry);">Failed to load history: ${err.message}</td></tr>`;
        }
    }

    if (filterEmotion) filterEmotion.addEventListener('change', loadHistory);
    if (filterSource) filterSource.addEventListener('change', loadHistory);

    if (btnClearHistory) {
        btnClearHistory.addEventListener('click', async () => {
            if (confirm('Are you sure you want to clear all emotion detection history?')) {
                await fetch('/api/history/delete/', { method: 'POST' });
                showToast('All detection records cleared!', '🧹');
                loadHistory();
            }
        });
    }

    if (btnExportCsv) {
        btnExportCsv.addEventListener('click', () => {
            window.location.href = '/api/export/?format=csv';
        });
    }

    if (btnExportJson) {
        btnExportJson.addEventListener('click', () => {
            window.location.href = '/api/export/?format=json';
        });
    }

    // --- Model Hub Cards Selection ---
    document.querySelectorAll('.btn-select-model').forEach(btn => {
        btn.addEventListener('click', () => {
            const filename = btn.dataset.filename;
            state.selectedModel = filename;
            if (modelSelector) modelSelector.value = filename;
            showToast(`Active model changed to: ${filename}`, '🧠');
            document.querySelectorAll('.btn-select-model').forEach(b => {
                b.textContent = 'Select Model';
                b.classList.remove('btn-primary');
                b.classList.add('btn-secondary');
            });
            btn.textContent = 'Active Model';
            btn.classList.remove('btn-secondary');
            btn.classList.add('btn-primary');
        });
    });

    console.log("Facial Expression Recognition Studio initialized.");
});
