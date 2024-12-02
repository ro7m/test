// Constants (updated to match Python implementation)
const DET_MEAN = [0.798, 0.785, 0.772];
const DET_STD = [0.264, 0.2749, 0.287];
const REC_MEAN = 0.694;
const REC_STD = 0.298;
const VOCAB = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~°£€¥¢฿àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ<eos><sos><pad>";

// DOM Elements (unchanged)
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

// Preprocessing functions
function preprocessImageForDetection(imageElement) {
    const targetSize = [1024, 1024];
    const tensor = tf.browser
        .fromPixels(imageElement)
        .resizeNearestNeighbor(targetSize)
        .toFloat()
        .transpose([2, 0, 1]) // Change to channel-first format
        .expandDims(0); // Add batch dimension

    // Normalize using mean and std
    const mean = tf.tensor(DET_MEAN.map(m => m * 255));
    const std = tf.tensor(DET_STD.map(s => s * 255));
    
    return tensor.sub(mean.reshape([1, 3, 1, 1])).div(std.reshape([1, 3, 1, 1]));
}

function preprocessImageForRecognition(crops) {
    const targetSize = [32, 128];
    const tensors = crops.map((crop) => {
        let tensor = tf.browser
            .fromPixels(crop)
            .resizeNearestNeighbor(targetSize)
            .toFloat()
            .transpose([2, 0, 1]) // Channel-first format
            .expandDims(0); // Add batch dimension

        // Normalize
        const mean = tf.scalar(255 * REC_MEAN);
        const std = tf.scalar(255 * REC_STD);
        
        return tensor.sub(mean).div(std);
    });

    // Concatenate if multiple crops
    return tensors.length > 1 ? tf.concat(tensors, 0) : tensors[0];
}

// Model loading function
async function loadModels() {
    try {
        const ort = window.ort; // ONNX Runtime

        // Load detection model (FAST)
        detectionModel = await ort.InferenceSession.create('models/rep_fast_base.onnx', {
            executionProviders: ['webgl', 'cpu']
        });

        // Load recognition model (PARSeq)
        recognitionModel = await ort.InferenceSession.create('models/parseq.onnx', {
            executionProviders: ['webgl', 'cpu']
        });

        console.log('ONNX Models loaded successfully');
    } catch (error) {
        console.error('Error loading ONNX models:', error);
    }
}

// Detection post-processing (similar to Python implementation)
function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function decodeText(logits) {
    // Similar to PARSeqPostProcessor in Python
    const vocabList = VOCAB.split('');
    
    // Find argmax indices
    const outIdxs = logits.map(row => row.indexOf(Math.max(...row)));
    
    // Convert indices to characters
    const wordValues = outIdxs.map(seq => 
        seq.map(idx => vocabList[idx])
            .join('')
            .split('<eos>')[0]
    );

    return wordValues[0]; // Return first prediction
}

async function detectAndRecognizeText(imageElement) {
    const ort = window.ort;
    const size = [1024, 1024];

    // Detect text regions
    const detectionInput = preprocessImageForDetection(imageElement);
    
    const detectionFeeds = {
        input: new ort.Tensor(
            'float32', 
            detectionInput.dataSync(), 
            detectionInput.shape
        )
    };

    // Run detection
    const detectionResults = await detectionModel.run(detectionFeeds);
    
    // Convert logits to probability map using sigmoid
    const probMap = Object.values(detectionResults)[0].data.map(sigmoid);
    
    // Extract bounding boxes (similar to previous implementation)
    const heatmapCanvas = document.createElement('canvas');
    heatmapCanvas.width = size[1];
    heatmapCanvas.height = size[0];
    const ctx = heatmapCanvas.getContext('2d');
    
    // Create ImageData from probability map
    const imageData = new ImageData(
        new Uint8ClampedArray(probMap.map(p => p * 255)),
        size[1],
        size[0]
    );
    ctx.putImageData(imageData, 0, 0);

    // Extract bounding boxes using OpenCV (unchanged)
    const boundingBoxes = extractBoundingBoxesFromHeatmap(heatmapCanvas, size);

    // Preview canvas setup
    previewCanvas.width = imageElement.width;
    previewCanvas.height = imageElement.height;
    const previewCtx = previewCanvas.getContext('2d');
    previewCtx.drawImage(imageElement, 0, 0);

    const crops = [];

    // Process detected regions
    for (const box of boundingBoxes) {
        // Draw bounding box
        const [x1, y1] = box.coordinates[0];
        const [x2, y2] = box.coordinates[2];
        const width = (x2 - x1) * imageElement.width;
        const height = (y2 - y1) * imageElement.height;
        const x = x1 * imageElement.width;
        const y = y1 * imageElement.height;

        previewCtx.strokeStyle = box.config.stroke;
        previewCtx.lineWidth = 2;
        previewCtx.strokeRect(x, y, width, height);

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

    // Recognize text in crops
    let fullText = '';
    const batchSize = 32;

    for (let i = 0; i < crops.length; i += batchSize) {
        const batch = crops.slice(i, i + batchSize);
        const inputTensor = preprocessImageForRecognition(batch);

        // Prepare input for recognition model
        const recognitionFeeds = {
            input: new ort.Tensor(
                'float32', 
                inputTensor.dataSync(), 
                inputTensor.shape
            )
        };

        // Run recognition
        const recognitionResults = await recognitionModel.run(recognitionFeeds);
        
        // Get logits and process
        const logits = Object.values(recognitionResults)[0].data;
        const reshapedLogits = [];
        for (let j = 0; j < logits.length; j += VOCAB.length) {
            reshapedLogits.push(logits.slice(j, j + VOCAB.length));
        }

        const words = decodeText(reshapedLogits);
        fullText += words + ' ';

        // Clean up
        inputTensor.dispose();
    }

    return fullText.trim();
}

// Existing helper functions (unchanged)
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

// Script loading functions
function loadONNXRuntime() {
    return new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.15.0/dist/onnx.min.js';
        script.onload = () => resolve();
        script.onerror = () => reject(new Error('Failed to load ONNX Runtime'));
        document.body.appendChild(script);
    });
}

// Modify init function to load ONNX Runtime
async function init() {
    await loadONNXRuntime(); // Load ONNX Runtime first
    await loadModels();
    await loadOpenCV();
    await setupCamera();
    captureButton.disabled = false;
    captureButton.textContent = 'Capture';
}

// Rest of the existing code (event listeners, camera setup, etc.) remains the same
