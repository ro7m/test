
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

  function extractBoundingBoxes(probMap, imageElement) {
    const binThresh = 0.5;
    const boxThresh = 0.5;
    const width = 1024;
    const height = 1024;
    const scaleX = imageElement.width / width;
    const scaleY = imageElement.height / height;

    // Binarize the probability map
    const binaryMap = probMap.map(val => val > binThresh ? 255 : 0);

    // Convert to 2D array for processing
    const bitmap = new Array(height).fill(null).map((_, y) => 
        binaryMap.slice(y * width, (y + 1) * width)
    );

    // Find contours using OpenCV.js methods
    const matBitmap = cv.matFromArray(height, width, cv.CV_8U, bitmap.flat());
    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    
    cv.findContours(matBitmap, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    const boxes = [];

    for (let i = 0; i < contours.size(); i++) {
        const contour = contours.get(i);
        
        // Create a bounding rectangle
        const rect = cv.boundingRect(contour);
        
        // Convert to points format for scoring
        const points = [
            [rect.x, rect.y],
            [rect.x, rect.y + rect.height],
            [rect.x + rect.width, rect.y + rect.height],
            [rect.x + rect.width, rect.y]
        ];

        // Compute box score similar to Python implementation
        const mask = new cv.Mat.zeros(height, width, cv.CV_8U);
        const contourPoints = cv.matFromArray(4, 1, cv.CV_32S, points.flat());
        cv.fillPoly(mask, [contourPoints], new cv.Scalar(255));
        
        const maskedPred = new cv.Mat();
        cv.bitwise_and(cv.matFromArray(height, width, cv.CV_32F, probMap), mask, maskedPred);
        const score = cv.mean(maskedPred)[0];

        // Filter boxes based on score and size
        if (score > boxThresh && 
            rect.width > 2 && rect.height > 2) {
            
            // Scale to original image dimensions and normalize
            const scaledBox = [
                (rect.x / width),
                (rect.y / height),
                ((rect.x + rect.width) / width),
                ((rect.y + rect.height) / height)
            ];

            boxes.push(scaledBox);
        }

        // Clean up OpenCV objects
        contour.delete();
        mask.delete();
        contourPoints.delete();
    }

    // Clean up additional OpenCV objects
    matBitmap.delete();
    contours.delete();
    hierarchy.delete();

    return boxes;
  }

        
    function computeBoxScore(pred, points) {
        // More sophisticated box score computation
        const scores = points.map(([x, y]) => pred[y * width + x]);
        const meanScore = scores.reduce((a, b) => a + b, 0) / scores.length;
        const variance = scores.reduce((sum, score) => 
            sum + Math.pow(score - meanScore, 2), 0) / scores.length;
        
        return {
            meanScore,
            variance
        };
    }

    function mergeNearbyBoxes(boxes, maxOverlap = 0.3) {
        // Sort boxes by area (largest first)
        boxes.sort((a, b) => {
            const areaA = (a[2] - a[0]) * (a[3] - a[1]);
            const areaB = (b[2] - b[0]) * (b[3] - b[1]);
            return areaB - areaA;
        });

        const mergedBoxes = [];
        const used = new Array(boxes.length).fill(false);

        for (let i = 0; i < boxes.length; i++) {
            if (used[i]) continue;

            let bestBox = boxes[i];
            used[i] = true;

            for (let j = i + 1; j < boxes.length; j++) {
                if (used[j]) continue;

                // Compute IoU (Intersection over Union)
                const xA = Math.max(bestBox[0], boxes[j][0]);
                const yA = Math.max(bestBox[1], boxes[j][1]);
                const xB = Math.min(bestBox[2], boxes[j][2]);
                const yB = Math.min(bestBox[3], boxes[j][3]);

                const intersectionArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
                const boxAArea = (bestBox[2] - bestBox[0]) * (bestBox[3] - bestBox[1]);
                const boxBArea = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1]);

                const iou = intersectionArea / (boxAArea + boxBArea - intersectionArea);

                if (iou > maxOverlap) {
                    // Merge boxes
                    bestBox = [
                        Math.min(bestBox[0], boxes[j][0]),
                        Math.min(bestBox[1], boxes[j][1]),
                        Math.max(bestBox[2], boxes[j][2]),
                        Math.max(bestBox[3], boxes[j][3])
                    ];
                    used[j] = true;
                }
            }

            mergedBoxes.push(bestBox);
        }

        return mergedBoxes;
    }

    const contours = findContours(bitmap);
    const boxes = contours.reduce((validBoxes, contour) => {
        // Filter out very small contours
        if (contour.length < 10) return validBoxes;

        // Compute bounding box and score
        const xs = contour.map(p => p[0]);
        const ys = contour.map(p => p[1]);
        const x1 = Math.min(...xs);
        const y1 = Math.min(...ys);
        const x2 = Math.max(...xs);
        const y2 = Math.max(...ys);

        // Compute box score
        const { meanScore, variance } = computeBoxScore(probMap, contour);
        
        // More strict filtering
        if (meanScore < boxThresh || variance > 0.5) return validBoxes;

        // Aspect ratio filtering to remove unlikely text regions
        const width = x2 - x1;
        const height = y2 - y1;
        const aspectRatio = width / height;
        if (aspectRatio < 0.1 || aspectRatio > 10) return validBoxes;

        // Scale to original image dimensions
        validBoxes.push([
            x1 * scaleX,
            y1 * scaleY,
            x2 * scaleX,
            y2 * scaleY
        ]);

        return validBoxes;
    }, []);
    
function decodeText(logits, vocab) {
    const sequences = [];
    const vocabLength = vocab.length;

    // Assuming logits is a 2D array of predictions for each crop
    for (let i = 0; i < logits.length / vocabLength; i++) {
        const seqLogits = logits.slice(i * vocabLength, (i + 1) * vocabLength);
        const sequence = seqLogits.map(sequenceLogits => {
            const maxIndex = sequenceLogits.indexOf(Math.max(...sequenceLogits));
            return vocab[maxIndex];
        }).join('').trim();
        sequences.push(sequence);
    }

    return sequences;
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

        /* for time being remove recognition

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
        */
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
    loadingMessageElement.textContent = '';
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
