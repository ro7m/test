
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
        detectionModel = await loadONNXModel('models/rep_fast_base.onnx', {
            executionProviders: ['webgl', 'wasm']
        });
        recognitionModel = await loadONNXModel('models/crnn_mobilenet_v3_large.onnx', {
            executionProviders: ['webgl', 'wasm']
        });
        
        console.log('ONNX Models loaded successfully');
        //showLoading('Ready to use ....');
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
    //showLoading('Setting up camera...');
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
                //hideLoading();
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

function preprocessImageForRecognition(crops, vocab, targetSize = [32, 128], mean = [0.694, 0.695, 0.693], std = [0.299, 0.296, 0.301]) {
    // Helper function to resize and pad image
    function resizeAndPadImage(imageData) {
        const canvas = new OffscreenCanvas(imageData.width, imageData.height);
        const ctx = canvas.getContext('2d');
        ctx.drawImage(imageData, 0, 0);

        const [targetHeight, targetWidth] = targetSize;
        let resizedWidth, resizedHeight;
        let aspectRatio = targetWidth / targetHeight;

        if (aspectRatio * imageData.height > imageData.width) {
            resizedHeight = targetHeight;
            resizedWidth = Math.round((targetHeight * imageData.width) / imageData.height);
        } else {
            resizedWidth = targetWidth;
            resizedHeight = Math.round((targetWidth * imageData.height) / imageData.width);
        }

        const resizeCanvas = new OffscreenCanvas(targetWidth, targetHeight);
        const resizeCtx = resizeCanvas.getContext('2d');
        
        resizeCtx.fillStyle = 'black';
        resizeCtx.fillRect(0, 0, targetWidth, targetHeight);

        const xOffset = Math.floor((targetWidth - resizedWidth) / 2);
        const yOffset = Math.floor((targetHeight - resizedHeight) / 2);

        resizeCtx.drawImage(
            canvas, 
            0, 0, imageData.width, imageData.height, 
            xOffset, yOffset, resizedWidth, resizedHeight
        );

        return resizeCtx.getImageData(0, 0, targetWidth, targetHeight);
    }

    // Process each crop
    const processedImages = crops.map(crop => {
        const resizedImage = resizeAndPadImage(crop);
        
        // Allocate a new Float32Array for the entire image (3 channels)
        const float32Data = new Float32Array(3 * targetSize[0] * targetSize[1]);
        
        // Normalize and separate channels
        for (let y = 0; y < targetSize[0]; y++) {
            for (let x = 0; x < targetSize[1]; x++) {
                const pixelIndex = (y * targetSize[1] + x) * 4;
                const channelSize = targetSize[0] * targetSize[1];
                
                // Extract RGB and normalize
                const r = (resizedImage.data[pixelIndex] / 255.0 - mean[0]) / std[0];
                const g = (resizedImage.data[pixelIndex + 1] / 255.0 - mean[1]) / std[1];
                const b = (resizedImage.data[pixelIndex + 2] / 255.0 - mean[2]) / std[2];

                // Store normalized values in float32Data
                float32Data[y * targetSize[1] + x] = r;
                float32Data[channelSize + y * targetSize[1] + x] = g;
                float32Data[2 * channelSize + y * targetSize[1] + x] = b;
            }
        }

        return float32Data;
    });

    // Concatenate multiple processed images
    if (processedImages.length > 1) {
        const combinedLength = 3 * targetSize[0] * targetSize[1] * processedImages.length;
        const combinedData = new Float32Array(combinedLength);
        
        processedImages.forEach((img, index) => {
            combinedData.set(img, index * img.length);
        });

        return {
            data: combinedData,
            dims: [processedImages.length, 3, targetSize[0], targetSize[1]]
        };
    }

    // Single image case
    return {
        data: processedImages[0],
        dims: [1, 3, targetSize[0], targetSize[1]]
    };
}     

async function recognizeText(crops, recognitionModel, vocab) {

     const preprocessedImage = preprocessImageForRecognition(
        crops.map(crop => crop.canvas)
    );

    // Create ONNX Runtime tensor
    const inputTensor = new ort.Tensor('float32', preprocessedImage.data, preprocessedImage.dims);

    // Run inference
    const feeds = { 'input': inputTensor };
    const results = await recognitionModel.run(feeds); 
  
       // Get logits
    const logits = results.logits.data;
    const [batchSize, height, numClasses] = results.logits.dims;

    // Softmax implementation
    function softmax(arr) {
        const maxVal = Math.max(...arr);
        const exp = arr.map(x => Math.exp(x - maxVal));
        const sumExp = exp.reduce((a, b) => a + b, 0);
        return exp.map(x => x / sumExp);
    }

    // Process logits
    const probabilities = [];
    for (let b = 0; b < batchSize; b++) {
        const batchProbs = [];
        for (let h = 0; h < height; h++) {
            // Extract logits for this position
            const positionLogits = [];
            for (let c = 0; c < numClasses; c++) {
                const index = b * (height * numClasses) + h * numClasses + c;
                positionLogits.push(logits[index]);
            }
            
            // Apply softmax to position
            batchProbs.push(softmax(positionLogits));
        }
        probabilities.push(batchProbs);
    }

    // Find argmax (best path)
    const bestPath = probabilities.map(batchProb => 
        batchProb.map(row => 
            row.indexOf(Math.max(...row))
        )
    );

    // Convert best path to text using vocab
    const decodedTexts = bestPath.map(sequence => 
        sequence
            .filter(idx => idx !== numClasses - 1)  // Remove blank token (last index)
            .map(idx => vocab[idx])
            .join('')
    );

    return {
        probabilities,
        bestPath,
        decodedTexts
    };
}

let startTime , endTime;

async function detectAndRecognizeText(imageElement) {

   startTime = performance.now(); 
   
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
    

        try {
        const results = await recognizeText(crops, recognitionModel, VOCAB);
        
        console.log('Decoded Texts:', results);        

       for (let i=0; i<results["decodedTexts"].length ; i++){

            extractedData.push({
                    word: results["decodedTexts"][i],
                    boundingBox: crops[i].bbox,
                    probablities: results["probabilities"][i]
                });
       } 
        endTime = performance.now();
        const totalTime = ((endTime - startTime)/1000).toFixed(2);
        resultTime.innerHTML += `<br><small> Processing Time: ${totalTime} seconds </small>`;   
      } catch (error) {
        console.error('Recognition error:', error);
   }
    
    return extractedData;
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
    startTime = null;
    endTime = null;
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
            extractedText = extractedData.map(item => item.word).join('<br>');
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
    
    //hideLoading();
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
