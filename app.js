
// Constants
const REC_MEAN = [0.694, 0.695, 0.693];
const REC_STD = [0.299, 0.296, 0.301];
const DET_MEAN = [0.798, 0.785, 0.772];
const DET_STD = [0.264, 0.2749, 0.287];
const VOCAB = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~°£€¥¢฿àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ";

const LOADING_STAGES = {
    INITIALIZING: 'Initializing application...',
    LOADING_MODELS: 'Loading detection and recognition models...',
    SETTING_UP_CAMERA: 'Setting up camera...',
    READY: 'Ready to capture',
};

const loadingMessageElement = document.getElementById('loadingMessage');

function createPerformanceTracker() {
    const startTimes = {};
    return {
        start: (label) => {
            startTimes[label] = performance.now();
        },
        end: (label) => {
            const endTime = performance.now();
            const duration = endTime - (startTimes[label] || endTime);
            console.log(`${label} took ${duration.toFixed(2)}ms`);
            return duration;
        }
    };
}
const performanceTracker = createPerformanceTracker();

function updateLoadingStage(stage) {
    console.log(`Stage: ${stage}`);
    if (loadingMessageElement) {
        loadingMessageElement.textContent = stage;
    }
}

// DOM Elements
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const previewCanvas = document.getElementById('previewCanvas');
const captureButton = document.getElementById('captureButton');
const confirmButton = document.getElementById('confirmButton');
const retryButton = document.getElementById('retryButton');
const actionButtons = document.getElementById('actionButtons');
const sendButton = document.getElementById('sendButton');
const discardButton = document.getElementById('discardButton');
const resultElement = document.getElementById('result');
const apiResponseElement = document.getElementById('apiResponse');

let imageDataUrl = '';
let extractedText = '';
let detectionModel;
let recognitionModel;

async function loadModels() {
    updateLoadingStage(LOADING_STAGES.LOADING_MODELS);
    performanceTracker.start('Model Loading');
    
    try {
        detectionModel = await ort.InferenceSession.create('models/rep_fast_base.onnx');
        recognitionModel = await ort.InferenceSession.create('models/parseq_dynamic.onnx');
        
        const modelLoadTime = performanceTracker.end('Model Loading');
        console.log(`Models loaded successfully in ${modelLoadTime.toFixed(2)}ms`);
    } catch (error) {
        console.error('Error loading models:', error);
        updateLoadingStage('Failed to load models. Please refresh.');
    }
}

async function setupCamera() {
    updateLoadingStage(LOADING_STAGES.SETTING_UP_CAMERA);
    performanceTracker.start('Camera Setup');
    
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                facingMode: 'environment',
                width: { ideal: 1280 },
                height: { ideal: 720 }
            } 
        });
        video.srcObject = stream;
        
        return new Promise((resolve, reject) => {
            video.onloadedmetadata = () => {
                const cameraSetupTime = performanceTracker.end('Camera Setup');
                console.log(`Camera setup completed in ${cameraSetupTime.toFixed(2)}ms`);
                updateLoadingStage(LOADING_STAGES.READY);
                resolve(video);
            };
            video.onerror = reject;
        });
    } catch (error) {
        console.error('Camera setup error:', error);
        updateLoadingStage('Failed to access camera. Please check permissions.');
        throw error;
    }
}

function preprocessImageForDetection(imageElement) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 1024;
    canvas.height = 1024;
    
    // Resize and center crop the image
    const scale = Math.max(1024 / imageElement.width, 1024 / imageElement.height);
    const scaledWidth = imageElement.width * scale;
    const scaledHeight = imageElement.height * scale;
    const offsetX = (1024 - scaledWidth) / 2;
    const offsetY = (1024 - scaledHeight) / 2;
    
    ctx.drawImage(imageElement, offsetX, offsetY, scaledWidth, scaledHeight);
    
    const imageData = ctx.getImageData(0, 0, 1024, 1024);
    const data = imageData.data;
    const tensor = new Float32Array(3 * 1024 * 1024);
    
    for (let i = 0; i < 1024 * 1024; i++) {
        const r = data[i * 4];
        const g = data[i * 4 + 1];
        const b = data[i * 4 + 2];
        
        tensor[i] = (r / 255 - DET_MEAN[0]) / DET_STD[0];
        tensor[i + 1024 * 1024] = (g / 255 - DET_MEAN[1]) / DET_STD[1];
        tensor[i + 2 * 1024 * 1024] = (b / 255 - DET_MEAN[2]) / DET_STD[2];
    }
    
    return new ort.Tensor('float32', tensor, [1, 3, 1024, 1024]);
}

function preprocessImageForRecognition(crops) {
    const processedCrops = crops.map(crop => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = 128;
        canvas.height = 32;
        
        // Resize maintaining aspect ratio
        const scale = Math.min(128 / crop.width, 32 / crop.height);
        const scaledWidth = crop.width * scale;
        const scaledHeight = crop.height * scale;
        const offsetX = (128 - scaledWidth) / 2;
        const offsetY = (32 - scaledHeight) / 2;
        
        ctx.drawImage(crop, offsetX, offsetY, scaledWidth, scaledHeight);
        
        const imageData = ctx.getImageData(0, 0, 128, 32);
        const data = imageData.data;
        const tensor = new Float32Array(3 * 32 * 128);
        
        for (let i = 0; i < 32 * 128; i++) {
            const r = data[i * 4];
            const g = data[i * 4 + 1];
            const b = data[i * 4 + 2];
            
            tensor[i] = (r / 255 - REC_MEAN[0]) / REC_STD[0];
            tensor[i + 32 * 128] = (g / 255 - REC_MEAN[1]) / REC_STD[1];
            tensor[i + 2 * 32 * 128] = (b / 255 - REC_MEAN[2]) / REC_STD[2];
        }
        
        return tensor;
    });

    const combinedTensor = new Float32Array(
        processedCrops.length * 3 * 32 * 128
    );
    processedCrops.forEach((crop, index) => {
        combinedTensor.set(
            crop, 
            index * (3 * 32 * 128)
        );
    });

    return new ort.Tensor(
        'float32', 
        combinedTensor, 
        [processedCrops.length, 3, 32, 128]
    );
}


async function detectAndRecognizeText(imageElement) {
    performanceTracker.start('Full OCR Process');
    updateLoadingStage('Processing image...');

    try {
        const detectionTensor = preprocessImageForDetection(imageElement);
        const detectionResults = await detectionModel.run({ input: detectionTensor });
        
        const probMap = Object.values(detectionResults)[0].data.map(val => 1 / (1 + Math.exp(-val)));
        
        const boundingBoxes = extractBoundingBoxes(probMap, imageElement);
        
        // Preview canvas drawing
        previewCanvas.width = imageElement.width;
        previewCanvas.height = imageElement.height;
        const ctx = previewCanvas.getContext('2d');
        ctx.drawImage(imageElement, 0, 0);

        const crops = [];
        
        // Draw and crop detected regions
        boundingBoxes.forEach((box, index) => {
            const [x1, y1, x2, y2] = box;
            
            // Draw bounding box
            ctx.strokeStyle = getRandomColor();
            ctx.lineWidth = 2;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

            const croppedCanvas = document.createElement('canvas');
            croppedCanvas.width = x2 - x1;
            croppedCanvas.height = y2 - y1;
            croppedCanvas.getContext('2d').drawImage(
                imageElement, 
                x1, y1, x2 - x1, y2 - y1,
                0, 0, x2 - x1, y2 - y1
            );

            crops.push(croppedCanvas);
        });

        // Process crops in batches
        const batchSize = 16;
        let fullText = '';

        for (let i = 0; i < crops.length; i += batchSize) {
            updateLoadingStage(`Processing text batch ${Math.floor(i/batchSize) + 1}...`);
            const batch = crops.slice(i, i + batchSize);
            const inputTensor = preprocessImageForRecognition(batch);

            const recognitionResults = await recognitionModel.run({ input: inputTensor });
            
            // Similar text decoding logic as before
            const logits = Object.values(recognitionResults)[0].data;
            const vocabLength = VOCAB.length;
            
            const batchTexts = [];
            for (let j = 0; j < batch.length; j++) {
                const sequenceLogits = logits.slice(
                    j * vocabLength, 
                    (j + 1) * vocabLength
                );
                const extractedWord = sequenceLogits
                    .map((_, index) => VOCAB[sequenceLogits.indexOf(Math.max(...sequenceLogits))])
                    .join('').trim();
                
                batchTexts.push(extractedWord);
            }
            
            fullText += batchTexts.join(' ') + ' ';
        }

        const totalOcrTime = performanceTracker.end('Full OCR Process');
        updateLoadingStage(`OCR completed in ${totalOcrTime.toFixed(2)}ms`);
        
        return fullText.trim();

    } catch (error) {
        console.error('OCR Processing Error:', error);
        updateLoadingStage('OCR processing failed. Please try again.');
        return '';
    }
}



function getRandomColor() {
    return '#' + Math.floor(Math.random()*16777215).toString(16);
}

function handleCapture() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    
    imageDataUrl = canvas.toDataURL('image/jpeg');
    resultElement.textContent = 'Processing image...';
    
    const img = new Image();
    img.src = imageDataUrl;
    img.onload = async () => {
        try {
            extractedText = await detectAndRecognizeText(img);
            resultElement.textContent = `Extracted Text: ${extractedText}`;
            
            // Show preview canvas and confirmation buttons
            previewCanvas.style.display = 'block';
            confirmButton.style.display = 'inline-block';
            retryButton.style.display = 'inline-block';
            captureButton.style.display = 'none';
        } catch (error) {
            console.error('Error during text extraction:', error);
            resultElement.textContent = 'Error occurred during text extraction';
        }
    };
}

function handleConfirm() {
    toggleButtons(true);
    previewCanvas.style.display = 'none';
    confirmButton.style.display = 'none';
    retryButton.style.display = 'none';
}

function handleRetry() {
    resetUI();
}

async function handleSend() {
    if (!extractedText) return;
    apiResponseElement.textContent = 'Submitting...';
    let msgKey = new Date().getTime();
    try {
        const response = await fetch('https://kvdb.io/NyKpFtJ7v392NS8ibLiofx/'+msgKey, {
            method: 'PUT',
            body: JSON.stringify({
                extractetAt: msgKey,
                data: extractedText,
                userId: "imageExt",
            }),
            headers: {
                'Content-type': 'application/json; charset=UTF-8',
            },
        });

        if (response.status !== 200) {
            throw new Error('Failed to push this data to server');
        } 
        
        apiResponseElement.textContent = 'Submitted the extract with ID : ' + msgKey; 
        
    } catch (error) {
        console.error('Error submitting to server:', error);
        apiResponseElement.textContent = 'Error occurred while submitting to server';
    } finally {
        resetUI();
    }
}

function toggleButtons(showActionButtons) {
    captureButton.style.display = showActionButtons ? 'none' : 'block';
    actionButtons.style.display = showActionButtons ? 'block' : 'none';
}

function resetUI() {
    toggleButtons(false);
    resultElement.textContent = '';
    apiResponseElement.textContent = '';
    imageDataUrl = '';
    extractedText = '';
    clearCanvas();
    previewCanvas.style.display = 'none';
    confirmButton.style.display = 'none';
    retryButton.style.display = 'none';
    captureButton.style.display = 'block';
}

function clearCanvas() {
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
    previewCanvas.getContext('2d').clearRect(0, 0, previewCanvas.width, previewCanvas.height);
}

async function init() {
    updateLoadingStage(LOADING_STAGES.INITIALIZING);
    performanceTracker.start('Total Initialization');

    try {
        await loadModels();
        await loadOpenCV();
        await setupCamera();
        
        captureButton.disabled = false;
        captureButton.textContent = 'Capture';
        
        const initTime = performanceTracker.end('Total Initialization');
        console.log(`Total initialization time: ${initTime.toFixed(2)}ms`);
    } catch (error) {
        console.error('Initialization failed:', error);
        updateLoadingStage('Initialization failed. Please refresh.');
    }
}

function loadOpenCV() {
    return new Promise((resolve) => {
        const script = document.createElement('script');
        script.src = 'https://docs.opencv.org/4.5.2/opencv.js';
        script.onload = () => resolve();
        document.body.appendChild(script);
    });
}

// Event Listeners
captureButton.addEventListener('click', handleCapture);
captureButton.addEventListener('touchstart', handleCapture);
confirmButton.addEventListener('click', handleConfirm);
confirmButton.addEventListener('touchstart', handleConfirm);
retryButton.addEventListener('click', handleRetry);
retryButton.addEventListener('touchstart', handleRetry);
sendButton.addEventListener('click', handleSend);
sendButton.addEventListener('touchstart', handleSend);
discardButton.addEventListener('click', resetUI);
discardButton.addEventListener('touchstart', resetUI);

// Initialize the application
init();

// Service Worker Registration
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('service-worker.js')
            .then(registration => {
                console.log('ServiceWorker registration successful with scope: ', registration.scope);
            }, err => {
                console.log('ServiceWorker registration failed: ', err);
            });
    });
}
