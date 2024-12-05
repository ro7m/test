// Constants
const REC_MEAN = 0.694;
const REC_STD = 0.298;
const DET_MEAN = 0.785;
const DET_STD = 0.275;
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
        // Load ONNX models
        const detectionModelArrayBuffer = await fetch('models/db_mobilenet_v2/model.onnx').then(r => r.arrayBuffer());
        const recognitionModelArrayBuffer = await fetch('models/crnn_mobilenet_v2/model.onnx').then(r => r.arrayBuffer());
        
        // Initialize ONNX inference session
        detectionModel = await onnx.InferenceSession.create(detectionModelArrayBuffer);
        recognitionModel = await onnx.InferenceSession.create(recognitionModelArrayBuffer);
        
        console.log('ONNX Models loaded successfully');
    } catch (error) {
        console.error('Error loading ONNX models:', error);
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
    canvas.width = 512;
    canvas.height = 512;
    
    // Resize and draw image
    ctx.drawImage(imageElement, 0, 0, 512, 512);
    
    // Get image data
    const imageData = ctx.getImageData(0, 0, 512, 512);
    const data = imageData.data;
    
    // Create input tensor (NCHW format)
    const input = new Float32Array(1 * 3 * 512 * 512);
    for (let i = 0; i < 512 * 512; i++) {
        const r = data[i * 4] / 255;
        const g = data[i * 4 + 1] / 255;
        const b = data[i * 4 + 2] / 255;
        
        input[i] = (r - DET_MEAN) / DET_STD;
        input[i + 512 * 512] = (g - DET_MEAN) / DET_STD;
        input[i + 2 * 512 * 512] = (b - DET_MEAN) / DET_STD;
    }
    
    return new onnx.Tensor(input, 'float32', [1, 3, 512, 512]);
}

function preprocessImageForRecognition(crops) {
    const targetSize = [32, 128];
    const inputTensors = [];

    for (const crop of crops) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = targetSize[1];
        canvas.height = targetSize[0];
        
        // Resize and draw image
        ctx.drawImage(crop, 0, 0, targetSize[1], targetSize[0]);
        
        // Get image data
        const imageData = ctx.getImageData(0, 0, targetSize[1], targetSize[0]);
        const data = imageData.data;
        
        // Create input tensor (NCHW format)
        const input = new Float32Array(3 * targetSize[0] * targetSize[1]);
        for (let i = 0; i < targetSize[0] * targetSize[1]; i++) {
            const r = data[i * 4] / 255;
            const g = data[i * 4 + 1] / 255;
            const b = data[i * 4 + 2] / 255;
            
            input[i] = (r - REC_MEAN) / REC_STD;
            input[i + targetSize[0] * targetSize[1]] = (g - REC_MEAN) / REC_STD;
            input[i + 2 * targetSize[0] * targetSize[1]] = (b - REC_MEAN) / REC_STD;
        }
        
        inputTensors.push(new onnx.Tensor(input, 'float32', [1, 3, targetSize[0], targetSize[1]]));
    }
    
    return inputTensors;
}

function decodeText(predictions) {
    const blank = 126;
    let fullText = '';

    for (const predictionTensor of predictions) {
        const prediction = predictionTensor.data;
        let collapsed = "";
        let lastChar = null;

        for (let i = 0; i < prediction.length; i++) {
            const k = prediction[i];
            if (k !== blank && k !== lastChar) {         
                collapsed += VOCAB[k];
                lastChar = k;
            } else if (k === blank) {
                lastChar = null;
            }
        }
        fullText += collapsed.trim() + ' ';
    }
    
    return fullText.trim();
}

async function getHeatMapFromImage(imageObject) {
    const inputTensor = preprocessImageForDetection(imageObject);
    
    const feeds = {};
    feeds[detectionModel.inputNames[0]] = inputTensor;
    
    const results = await detectionModel.run(feeds);
    const outputTensor = results[detectionModel.outputNames[0]];
    
    // Convert ONNX tensor to canvas
    const heatmapCanvas = document.createElement('canvas');
    heatmapCanvas.width = imageObject.width;
    heatmapCanvas.height = imageObject.height;
    const ctx = heatmapCanvas.getContext('2d');
    
    const imageData = ctx.createImageData(heatmapCanvas.width, heatmapCanvas.height);
    const tensorData = outputTensor.data;
    
    for (let i = 0; i < tensorData.length; i++) {
        const value = Math.max(0, Math.min(255, tensorData[i] * 255));
        imageData.data[i * 4] = value;     // R
        imageData.data[i * 4 + 1] = value; // G
        imageData.data[i * 4 + 2] = value; // B
        imageData.data[i * 4 + 3] = 255;   // A
    }
    
    ctx.putImageData(imageData, 0, 0);
    return heatmapCanvas;
}

// Rest of the previous implementation remains the same, with modifications to use ONNX runtime
// (extractBoundingBoxesFromHeatmap, handleCapture, detectAndRecognizeText, etc.)

async function detectAndRecognizeText(imageElement) {
    const size = [512, 512];
    const heatmapCanvas = await getHeatMapFromImage(imageElement);
    const boundingBoxes = extractBoundingBoxesFromHeatmap(heatmapCanvas, size);
    console.log('extractBoundingBoxesFromHeatmap', boundingBoxes);

    previewCanvas.width = imageElement.width;
    previewCanvas.height = imageElement.height;
    const ctx = previewCanvas.getContext('2d');
    ctx.drawImage(imageElement, 0, 0);

    const crops = [];

    for (const box of boundingBoxes) {
        // Draw bounding box
        const [x1, y1] = box.coordinates[0];
        const [x2, y2] = box.coordinates[2];
        const width = (x2 - x1) * imageElement.width;
        const height = (y2 - y1) * imageElement.height;
        const x = x1 * imageElement.width;
        const y = y1 * imageElement.height;

        ctx.strokeStyle = box.config.stroke;
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, width, height);

        // Create crop
        const croppedCanvas = document.createElement('canvas');
        croppedCanvas.width = width;
        croppedCanvas.height = height;
        croppedCanvas.getContext('2d').drawImage(
            imageElement, 
            x, y, width, height,
            0, 0, width, height
        );

        crops.push(croppedCanvas);
    }

    // Process crops in batches
    const batchSize = 32;
    let fullText = '';
    for (let i = 0; i < crops.length; i += batchSize) {
        const batch = crops.slice(i, i + batchSize);
        const inputTensors = preprocessImageForRecognition(batch);
        
        const predictions = [];
        for (const inputTensor of inputTensors) {
            const feeds = {};
            feeds[recognitionModel.inputNames[0]] = inputTensor;
            
            const results = await recognitionModel.run(feeds);
            const outputTensor = results[recognitionModel.outputNames[0]];
            
            // Argmax equivalent for ONNX tensor
            const argmaxPrediction = new Float32Array(outputTensor.dims[0] * outputTensor.dims[1]);
            for (let j = 0; j < outputTensor.dims[0]; j++) {
                let maxIndex = 0;
                let maxValue = outputTensor.data[j * outputTensor.dims[1]];
                for (let k = 1; k < outputTensor.dims[1]; k++) {
                    if (outputTensor.data[j * outputTensor.dims[1] + k] > maxValue) {
                        maxValue = outputTensor.data[j * outputTensor.dims[1] + k];
                        maxIndex = k;
                    }
                }
                argmaxPrediction[j] = maxIndex;
            }
            
            predictions.push(new onnx.Tensor(argmaxPrediction, 'float32', outputTensor.dims));
        }
        
        const words = decodeText(predictions);
        fullText += words + ' ';
    }
    
    return fullText.trim();
}

// Remaining initialization and event listeners stay the same
async function init() {
    // Add ONNX.js script loading
    await loadONNXRuntime();
    
    await loadModels();
    await loadOpenCV();
    await setupCamera();
    captureButton.disabled = false;
    captureButton.textContent = 'Capture';
}

function loadONNXRuntime() {
    return new Promise((resolve) => {
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/onnxjs/dist/onnx.min.js';
        script.onload = () => resolve();
        document.body.appendChild(script);
    });
}

// Rest of the code remains the same
