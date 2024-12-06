
// Constants
const REC_MEAN = [0.694, 0.695, 0.693];
const REC_STD = [0.299, 0.296, 0.301];
const DET_MEAN = [0.798, 0.785, 0.772];
const DET_STD = [0.264, 0.2749, 0.287];
const VOCAB = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~°£€¥¢฿àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ";

const TARGET_SIZE = [1024, 1024];

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
const loadingIndicator = document.getElementById('loadingIndicator');
const appContainer = document.getElementById('appContainer');

let modelLoadingPromise;
let imageDataUrl = '';
let extractedText = '';
let extractedData = [];
let detectionModel;
let recognitionModel;


function showLoading(message) {
    loadingIndicator.textContent = message;
    loadingIndicator.style.display = 'block';
    //appContainer.style.display = 'none';
}

function hideLoading() {
    loadingIndicator.style.display = 'none';
    //appContainer.style.display = 'block';
}

// ONNX Runtime Web
async function loadONNXModel(modelPath) {
    showLoading(`Loading ONNX model from ${modelPath}...`);
    try {
        const response = await fetch(modelPath);
        const model = await response.arrayBuffer();
        return await ort.InferenceSession.create(model);
    } catch (error) {
        console.error('Error loading ONNX model:', error);
        showLoading('Error loading models. Please refresh the page.');
        throw error;
    }
}

async function loadModels() {
    try {
        detectionModel = await loadONNXModel('models/rep_fast_base.onnx');
        recognitionModel = await loadONNXModel('models/parseq_dynamic.onnx');
        
        console.log('ONNX Models loaded successfully');
        hideLoading();
    } catch (error) {
        console.error('Error loading models:', error);
        showLoading('Error loading models. Please refresh the page.');
    }
}

function initializeModelLoading() {
    modelLoadingPromise = loadModels();
}

async function ensureModelsLoaded() {
    if (modelLoadingPromise) {
        await modelLoadingPromise;
    }
}

async function setupCamera() {
    showLoading('Setting up camera...');
    try {
        const constraints = {
            video: {
                facingMode: 'environment',
                width: { ideal: 1024 },
                height: { ideal: 1024 },
                focusMode: 'continuous',
                advanced: [{
                    focusMode: 'continuous'
                }]
            },
            audio: false
        };

        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        
        // Get video track to apply additional constraints
        const videoTrack = stream.getVideoTracks()[0];
        if (videoTrack) {
            try {
                const capabilities = videoTrack.getCapabilities();
                // Only apply focus settings if the device supports them
                if (capabilities.focusMode) {
                    await videoTrack.applyConstraints({
                        focusMode: 'continuous'
                    });
                }
            } catch (error) {
                console.warn('Could not apply focus constraints:', error);
                // Continue even if focus settings fail
            }
        }

        video.srcObject = stream;
        return new Promise((resolve) => {
            video.onloadedmetadata = () => {
                hideLoading();
                resolve(video);
            };
        });
    } catch (error) {
        console.error('Error setting up camera:', error);
        showLoading('Error setting up camera. Please check permissions and refresh.');
        throw error;
    }
}

function triggerFocus() {
    if (video.srcObject) {
        const videoTrack = video.srcObject.getVideoTracks()[0];
        if (videoTrack) {
            const capabilities = videoTrack.getCapabilities();
            if (capabilities.focusMode) {
                try {
                    videoTrack.applyConstraints({
                        focusMode: 'continuous'
                    });
                } catch (error) {
                    console.warn('Could not trigger focus:', error);
                }
            }
        }
    }
}

function preprocessImageForDetection(imageElement) {
    const canvas = document.createElement('canvas');
    canvas.width = TARGET_SIZE[0];
    canvas.height = TARGET_SIZE[1];
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imageElement, 0, 0, TARGET_SIZE[0], TARGET_SIZE[1]);

    const imageData = ctx.getImageData(0, 0, TARGET_SIZE[0], TARGET_SIZE[1]);
    const data = imageData.data;
    const preprocessedData = new Float32Array(TARGET_SIZE[0] * TARGET_SIZE[1] * 3);

    for (let i = 0; i < data.length; i += 4) {
        const idx = i / 4;
        preprocessedData[idx] = (data[i] / 255 - DET_MEAN[0]) / DET_STD[0];       // R
        preprocessedData[idx + TARGET_SIZE[0] * TARGET_SIZE[1]] = (data[i + 1] / 255 - DET_MEAN[1]) / DET_STD[1];  // G
        preprocessedData[idx + TARGET_SIZE[0] * TARGET_SIZE[1] * 2] = (data[i + 2] / 255 - DET_MEAN[2]) / DET_STD[2];  // B
    }

    return preprocessedData;
}

async function detectText(imageObject, returnModelOutput = false) {
    const inputTensor = preprocessImageForDetection(imageObject);
    const feeds = {
        input: new ort.Tensor('float32', inputTensor, [1, 3, TARGET_SIZE[0], TARGET_SIZE[1]])
    };

    const detectionResults = await detectionModel.run(feeds);
    //const logitsData = logits.logits.data; // Adjust based on actual model output
    const probMap = Object.values(detectionResults)[0].data.map(val => 1 / (1 + Math.exp(-val)));

    const out = {};

    if (returnModelOutput) {
        out.out_map = probMap;
    }

    out.preds = postprocessProbabilityMap(probMap);

    return out;
}

function postprocessProbabilityMap(probMap) {
    const threshold = 0.1; // Adjust based on your specific requirements
    return probMap.map(prob => prob > threshold ? 1 : 0);
}

function createHeatmapFromProbMap(probMap) {
    const heatmapCanvas = document.createElement('canvas');
    heatmapCanvas.width = TARGET_SIZE[0];
    heatmapCanvas.height = TARGET_SIZE[1];
    const ctx = heatmapCanvas.getContext('2d');
    const imageData = ctx.createImageData(TARGET_SIZE[0], TARGET_SIZE[1]);
    
    probMap.forEach((prob, i) => {
        const pixelValue = Math.round(prob * 255);
        imageData.data[i * 4] = pixelValue;     // R
        imageData.data[i * 4 + 1] = pixelValue; // G
        imageData.data[i * 4 + 2] = pixelValue; // B
        imageData.data[i * 4 + 3] = 255;        // A
    });
    
    ctx.putImageData(imageData, 0, 0);
    return heatmapCanvas;
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

function decodeText(bestPath) {
    const blank = 126;
    let collapsed = "";
    let lastChar = null;

    for (const sequence of bestPath) {
        const values = sequence.data;
        for (const k of values) {
            if (k !== blank && k !== lastChar) {         
                collapsed += VOCAB[k];
                lastChar = k;
            } else if (k === blank) {
                lastChar = null;
            }
        }
        collapsed += ' ';
    }
    return collapsed.trim();
}

async function detectAndRecognizeText(imageElement) {
   
   const detectionResult = await detectText(imageElement, true);
        
   const heatmapCanvas = createHeatmapFromProbMap(detectionResult.out_map);
        
   const boundingBoxes = extractBoundingBoxesFromHeatmap(heatmapCanvas, TARGET_SIZE);

    previewCanvas.width = TARGET_SIZE[0];
    previewCanvas.height = TARGET_SIZE[1];
    const ctx = previewCanvas.getContext('2d');
    ctx.drawImage(imageElement, 0, 0);

    const crops = [];
    
    for (const box of boundingBoxes) {
        const [x1, y1] = box.coordinates[0];
        const [x2, y2] = box.coordinates[2];
        const width = (x2 - x1) * imageElement.width;
        const height = (y2 - y1) * imageElement.height;
        const x = x1 * imageElement.width;
        const y = y1 * imageElement.height;

        ctx.strokeStyle = box.config.stroke;
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, width, height);

        const croppedCanvas = document.createElement('canvas');
        croppedCanvas.width = Math.min(width, 128);
        croppedCanvas.height = Math.min(height, 32);
        croppedCanvas.getContext('2d').drawImage(
            imageElement, 
            x, y, width, height,
            0, 0, width, height
        );

        crops.push({
            canvas: croppedCanvas,
            bbox: {
                x: Math.round(x),
                y: Math.round(y),
                width: Math.round(width),
                height: Math.round(height)
            }
        });
    }

    const batchSize = 32;
    let fullText = '';
    
    for (let i = 0; i < crops.length; i += batchSize) {
        const batch = crops.slice(i, i + batchSize);
        const inputTensor = preprocessImageForRecognition(batch.map(crop => crop.canvas));

        const recognitionResults = await recognitionModel.run({ input: inputTensor });
        const logits = Object.values(recognitionResults)[0].data;
        const vocabLength = VOCAB.length;

        const batchTexts = [];
        for (let j = 0; j < batch.length; j++) {
            // Reshape logits for the current sequence
            const sequenceLogits = logits.slice(
                j * vocabLength, 
                (j + 1) * vocabLength
            );

            // Find indices of max probabilities
            const maxIndex = sequenceLogits.reduce(
                (maxIdx, val, idx, arr) => val > arr[maxIdx] ? idx : maxIdx, 
                0
            );

            const extractedWord = VOCAB[maxIndex] || '';

            batchTexts.push(extractedWord);

            extractedData.push({
                word: extractedWord,
                boundingBox: batch[j].bbox
            });
        }
        
        fullText += batchTexts.join(' ') + ' ';
    }          

    return extractedData;
}

function decodeText(bestPath) {
    const blank = 126;
    let collapsed = "";
    let lastChar = null;

    for (const sequence of bestPath) {
        const values = sequence.dataSync();
        for (const k of values) {
            if (k !== blank && k !== lastChar) {         
                collapsed += VOCAB[k];
                lastChar = k;
            } else if (k === blank) {
                lastChar = null;
            }
        }
        collapsed += ' ';
    }
    return collapsed.trim();
}

function clamp(number, size) {
    return Math.max(0, Math.min(number, size));
}

function transformBoundingBox(contour, id, size) {
    let offset = (contour.width * contour.height * 1.8) / (2 * (contour.width + contour.height));
    const p1 = clamp(contour.x - offset, size[1]) - 1;
    const p2 = clamp(p1 + contour.width + 2 * offset, size[1]) - 1;
    const p3 = clamp(contour.y - offset, size[0]) - 1;
    const p4 = clamp(p3 + contour.height + 2 * offset, size[0]) - 1;
    return {
        id,
        config: {
            stroke: getRandomColor(),
        },
        coordinates: [
            [p1 / size[1], p3 / size[0]],
            [p2 / size[1], p3 / size[0]],
            [p2 / size[1], p4 / size[0]],
            [p1 / size[1], p4 / size[0]],
        ],
    };
}

function extractBoundingBoxesFromHeatmap(heatmapCanvas, size) {
    let src = cv.imread(heatmapCanvas);
    cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
    cv.threshold(src, src, 77, 255, cv.THRESH_BINARY);
    cv.morphologyEx(src, src, cv.MORPH_OPEN, cv.Mat.ones(2, 2, cv.CV_8U));
    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat();
    cv.findContours(src, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    
    const boundingBoxes = [];
    for (let i = 0; i < contours.size(); ++i) {
        const contourBoundingBox = cv.boundingRect(contours.get(i));
        if (contourBoundingBox.width > 2 && contourBoundingBox.height > 2) {
            boundingBoxes.unshift(transformBoundingBox(contourBoundingBox, i, size));
        }
    }
    
    src.delete();
    contours.delete();
    hierarchy.delete();
    return boundingBoxes;
    }



function getRandomColor() {
    return '#' + Math.floor(Math.random()*16777215).toString(16);
}

function disableCaptureButton() {
    captureButton.disabled = true;
    captureButton.textContent = 'Processing...';
}

function enableCaptureButton() {
    captureButton.disabled = false;
    captureButton.textContent = 'Capture';
}


async function handleCapture() {
    disableCaptureButton();
    showLoading('Processing image...');

    await ensureModelsLoaded();  // Ensure models are loaded before processing

    const targetSize = TARGET_SIZE;
    canvas.width = targetSize[0];
    canvas.height = targetSize[1];
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);

    imageDataUrl = canvas.toDataURL('image/jpeg');
    
    const img = new Image();
    img.src = imageDataUrl;
    img.onload = async () => {
        try {
            extractedData = await detectAndRecognizeText(img);
            extractedText = extractedData.map(item => item.word).join(' ');
            resultElement.textContent = `Extracted Text: ${extractedText}`;
            
            previewCanvas.style.display = 'block';
            confirmButton.style.display = 'inline-block';
            retryButton.style.display = 'inline-block';
            captureButton.style.display = 'none';
        } catch (error) {
            console.error('Error during text extraction:', error);
            resultElement.textContent = 'Error occurred during text extraction';
        } finally {
            enableCaptureButton();
            hideLoading();
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
                probableTextContent: extractedText,
                boundingBoxes: extractedData,
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
    extractedData = [];
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
    showLoading('Initializing...');
    
    initializeModelLoading();
    await setupCamera();
    
    captureButton.disabled = false;
    captureButton.textContent = 'Capture';
    
    hideLoading();
}

// Event Listeners
captureButton.addEventListener('click', handleCapture);
captureButton.addEventListener('touchstart', () => {
    triggerFocus();
    // Wait a short moment for focus to adjust
    setTimeout(handleCapture, 500);
});
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

let deferredPrompt;
const installBtn = document.getElementById('install-btn');

window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    deferredPrompt = e;
    installBtn.style.display = 'block';

    installBtn.addEventListener('click', (e) => {
        installBtn.style.display = 'none';
        deferredPrompt.prompt();
        deferredPrompt.userChoice.then((choiceResult) => {
            if (choiceResult.outcome === 'accepted') {
                console.log('User accepted the A2HS prompt');
            } else {
                console.log('User dismissed the A2HS prompt');
            }
            deferredPrompt = null;
        });
    });
});

window.addEventListener('appinstalled', (evt) => {
    console.log('App was installed.');
    installBtn.style.display = 'none';
});
