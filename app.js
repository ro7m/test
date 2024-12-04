
// Constants
const REC_MEAN = [0.694, 0.695, 0.693];
const REC_STD = [0.299, 0.296, 0.301];
const DET_MEAN = [0.798, 0.785, 0.772];
const DET_STD = [0.264, 0.2749, 0.287];
const VOCAB = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~°£€¥¢฿àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ";

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
    try {
        detectionModel = await ort.InferenceSession.create('models/rep_fast_base.onnx');
        recognitionModel = await ort.InferenceSession.create('models/parseq_dynamic.onnx');
        console.log('Models loaded successfully');
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
            facingMode: 'environment',
            width: { ideal: 1280 },
            height: { ideal: 720 }
        } 
    });
    video.srcObject = stream;
    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            resolve(video);
        };
    });
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

function decodeText(predictions) {
    return predictions.map(([word, _]) => word).join(' ');
}

async function detectAndRecognizeText(imageElement) {
    const detectionTensor = preprocessImageForDetection(imageElement);
    
    const detectionFeed = {
        input: detectionTensor
    };

    const detectionResults = await detectionModel.run(detectionFeed);
    const detectionOutput = detectionResults.output.data;

    // Threshold and binarize using sigmoid
    const probMap = detectionOutput.map(val => 1 / (1 + Math.exp(-val)));

    // Extract bounding boxes (simplified version)
    const boundingBoxes = extractBoundingBoxes(probMap, imageElement);

    previewCanvas.width = imageElement.width;
    previewCanvas.height = imageElement.height;
    const ctx = previewCanvas.getContext('2d');
    ctx.drawImage(imageElement, 0, 0);

    const crops = [];
    const drawnBoxes = [];

    for (const box of boundingBoxes) {
        const [x1, y1, x2, y2] = box;
        
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
        drawnBoxes.push(box);
    }

    let fullText = '';
    const batchSize = 32;

    for (let i = 0; i < crops.length; i += batchSize) {
        const batch = crops.slice(i, i + batchSize);
        const inputTensor = preprocessImageForRecognition(batch);

        const recognitionFeed = {
            input: inputTensor
        };

        const recognitionResults = await recognitionModel.run(recognitionFeed);
        const logits = recognitionResults.output.data;

        // Replicate the PARSeq post-processing
        const wordResults = [];
        const seqLength = inputTensor.dims[0];
        const vocabLength = 127; // Adjust based on actual vocab length 

        for (let j = 0; j < seqLength; j++) {
            const sequenceLogits = logits.slice(
                j * vocabLength, 
                (j + 1) * vocabLength
            );
            const maxIndex = sequenceLogits.indexOf(
                Math.max(...sequenceLogits)
            );
            const word = VOCAB[maxIndex];
            wordResults.push(word);
        }

        const extractedBatchText = wordResults.join('');
        fullText += extractedBatchText + ' ';
    }

    return fullText.trim();
}

function extractBoundingBoxes(probMap, imageElement) {
    // Simplified bounding box extraction
    // In a real implementation, you'd use more sophisticated contour detection
    const threshold = 0.5;
    const boxes = [];
    const width = 1024;
    const height = 1024;
    const scaleX = imageElement.width / width;
    const scaleY = imageElement.height / height;

    for (let y = 0; y < height; y += 10) {
        for (let x = 0; x < width; x += 10) {
            const index = y * width + x;
            if (probMap[index] > threshold) {
                const searchBoxSize = 50;
                const x1 = Math.max(0, x - searchBoxSize / 2);
                const y1 = Math.max(0, y - searchBoxSize / 2);
                const x2 = Math.min(width, x + searchBoxSize / 2);
                const y2 = Math.min(height, y + searchBoxSize / 2);

                boxes.push([
                    x1 * scaleX,
                    y1 * scaleY,
                    x2 * scaleX,
                    y2 * scaleY
                ]);
            }
        }
    }

    return boxes;
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
    await loadModels();
    await loadOpenCV();
    await setupCamera();
    captureButton.disabled = false;
    captureButton.textContent = 'Capture';
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
