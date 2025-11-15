// Configuration
const API_URL = 'http://localhost:5000';
const PREDICTION_INTERVAL = 1500; // Predict every 1.5 seconds
const MIN_CONFIDENCE = 0.60;

// State
let webcamStream = null;
let sessionActive = false;
let predictionInterval = null;
let recognizedSigns = [];
let lastPredictedLetter = null;

// DOM Elements
const webcam = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const sessionBtn = document.getElementById('sessionBtn');
const clearBtn = document.getElementById('clearBtn');
const glossesDisplay = document.getElementById('glossesDisplay');
const sentenceText = document.getElementById('sentenceText');
const statusText = document.getElementById('statusText');
const confidenceText = document.getElementById('confidenceText');
const cameraOverlay = document.getElementById('cameraOverlay');

// Initialize webcam
async function initWebcam() {
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 }
        });
        webcam.srcObject = webcamStream;
        canvas.width = 640;
        canvas.height = 480;
        cameraOverlay.classList.add('hidden');
        updateStatus('Camera ready', 'success');
        return true;
    } catch (error) {
        console.error('Error accessing webcam:', error);
        updateStatus('Camera access denied', 'error');
        alert('Please allow camera access to use this application.');
        return false;
    }
}

// Stop webcam
function stopWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
        cameraOverlay.classList.remove('hidden');
    }
}

// Capture frame from webcam
function captureFrame() {
    ctx.drawImage(webcam, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.8);
}

// Send frame to backend for prediction
async function predictSign() {
    if (!sessionActive) return;

    try {
        const imageData = captureFrame();
        
        updateStatus('Predicting...', 'processing');

        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        });

        if (!response.ok) {
            throw new Error('Prediction failed');
        }

        const result = await response.json();
        
        if (result.success) {
            handlePrediction(result);
        } else {
            updateStatus('Prediction error', 'error');
        }
    } catch (error) {
        console.error('Prediction error:', error);
        updateStatus('Connection error', 'error');
    }
}

// Handle prediction result
function handlePrediction(result) {
    const { letter, confidence, below_threshold } = result;
    
    // Update confidence display
    confidenceText.textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;
    
    if (below_threshold) {
        updateStatus('Low confidence - try again', 'warning');
        return;
    }
    
    // Special handling for "nothing" - don't add to glosses
    if (letter.toLowerCase() === 'nothing') {
        updateStatus('Nothing detected', 'info');
        lastPredictedLetter = letter;
        return;
    }
    
    // Avoid duplicate consecutive predictions
    if (letter === lastPredictedLetter) {
        updateStatus(`Detected: ${letter} (duplicate skipped)`, 'success');
        return;
    }
    
    // Add sign to glosses
    lastPredictedLetter = letter;
    
    // Special handling for "space" - add actual space, not the word
    if (letter.toLowerCase() === 'space') {
        recognizedSigns.push(' ');
        updateStatus('Detected: SPACE', 'success');
    } else {
        recognizedSigns.push(letter);
        updateStatus(`Detected: ${letter}`, 'success');
    }
    
    updateGlossesDisplay();
    updateSentence();
}

// Update glosses display
function updateGlossesDisplay() {
    glossesDisplay.innerHTML = '';
    
    recognizedSigns.forEach((sign, index) => {
        const glossItem = document.createElement('div');
        glossItem.className = sign === '?' ? 'gloss-item unrecognized' : 'gloss-item';
        glossItem.textContent = sign;
        glossesDisplay.appendChild(glossItem);
    });
    
    if (recognizedSigns.length === 0) {
        glossesDisplay.innerHTML = '<div style="color: #a0aec0; font-size: 14px;">Signs will appear here...</div>';
    }
}

// Update sentence display
function updateSentence() {
    if (recognizedSigns.length === 0) {
        sentenceText.textContent = 'Start signing to see translation...';
        return;
    }
    
    // Filter out unrecognized signs for sentence
    const validSigns = recognizedSigns.filter(sign => sign !== '?');
    
    if (validSigns.length === 0) {
        sentenceText.textContent = 'No valid signs recognized yet...';
        return;
    }
    
    // Simple sentence formation (you can make this more sophisticated)
    const sentence = validSigns.join(' ').toLowerCase();
    sentenceText.textContent = sentence.charAt(0).toUpperCase() + sentence.slice(1) + '.';
}

// Start/Stop session
async function toggleSession() {
    if (!sessionActive) {
        // Start session
        const cameraReady = await initWebcam();
        if (!cameraReady) return;
        
        sessionActive = true;
        sessionBtn.classList.add('active');
        sessionBtn.querySelector('.btn-text').textContent = 'Stop Session';
        sessionBtn.querySelector('.btn-icon').textContent = '⏹';
        
        updateStatus('Session active - start signing!', 'success');
        
        // Start prediction loop
        predictionInterval = setInterval(predictSign, PREDICTION_INTERVAL);
    } else {
        // Stop session
        sessionActive = false;
        sessionBtn.classList.remove('active');
        sessionBtn.querySelector('.btn-text').textContent = 'Start Session';
        sessionBtn.querySelector('.btn-icon').textContent = '▶';
        
        clearInterval(predictionInterval);
        stopWebcam();
        
        updateStatus('Session stopped', 'info');
        confidenceText.textContent = '';
    }
}

// Clear all signs
function clearAllSigns() {
    recognizedSigns = [];
    lastPredictedLetter = null;
    updateGlossesDisplay();
    updateSentence();
    updateStatus('Cleared all signs', 'info');
    confidenceText.textContent = '';
}

// Update status message
function updateStatus(message, type = 'info') {
    statusText.textContent = message;
    
    // Color coding based on status type
    const colors = {
        success: '#48bb78',
        error: '#e53e3e',
        warning: '#ed8936',
        processing: '#4299e1',
        info: '#a0aec0'
    };
    
    statusText.style.color = colors[type] || colors.info;
}

// Check backend connection
async function checkBackendConnection() {
    try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        
        if (data.status === 'healthy') {
            console.log('✓ Backend connected successfully');
            console.log(`  Model loaded with ${data.num_classes} classes`);
            return true;
        }
    } catch (error) {
        console.error('✗ Backend connection failed:', error);
        alert('Cannot connect to backend server. Please ensure Flask server is running on http://localhost:5000');
        return false;
    }
}

// Event listeners
sessionBtn.addEventListener('click', toggleSession);
clearBtn.addEventListener('click', clearAllSigns);

// Initialize app
document.addEventListener('DOMContentLoaded', async () => {
    console.log('ISL Connect - Initializing...');
    
    // Check backend connection
    const connected = await checkBackendConnection();
    
    if (connected) {
        updateStatus('Ready to start', 'success');
    } else {
        updateStatus('Backend not connected', 'error');
    }
    
    // Initialize displays
    updateGlossesDisplay();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (webcamStream) {
        stopWebcam();
    }
});