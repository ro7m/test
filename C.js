// Constants
const REC_MEAN = 0.694;
const REC_STD = 0.298;
const DET_MEAN = 0.785;
const DET_STD = 0.275;
const VOCAB = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~°£€¥¢฿àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ";
const TARGET_SIZE = [1024, 1024];

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
const loadingIndicator = document.getElementById('loadingIndicator');
const appContainer = document.getElementById('appContainer');

let modelLoadingPromise;
let imageDataUrl = '';
let extractedText = '';
let extractedData = [];
let detectionModel;
let recognitionModel;

// ONNX Runtime Web
async function loadONNXModel(modelPath) {
    showLoading(`Loading ONNX model from ${modelPath}...`);
    try {
        const response = await fetch(modelPath);
        const model = await response.arrayBuffer();
        return await ort.InferenceSession.create(model, { executionProviders: ['webgl'] });
    } catch (error) {
        console.error('Error loading ONNX model:', error);
        showLoading('Error loading models. Please refresh the page.');
        throw error;
    }
}

async function loadModels() {
    try {
        detectionModel = await loadONNXModel('models/db_mobilenet_v2/model.onnx');
        recognitionModel = await loadONNXModel('models/crnn_mobilenet_v2/model.onnx');
        
        console.log('ONNX Models loaded successfully');
        hideLoading();
    } catch (error) {
        console.error('Error loading models:', error);
        showLoading('Error loading models. Please refresh the page.');
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
        preprocessedData[idx] = (data[i] / 255 - DET_MEAN) / DET_STD;       // R
        preprocessedData[idx + TARGET_SIZE[0] * TARGET_SIZE[1]] = (data[i + 1] / 255 - DET_MEAN) / DET_STD;  // G
        preprocessedData[idx + TARGET_SIZE[0] * TARGET_SIZE[1] * 2] = (data[i + 2] / 255 - DET_MEAN) / DET_STD;  // B
    }

    return preprocessedData;
}

async function getHeatMapFromImage(imageObject) {
    const inputTensor = preprocessImageForDetection(imageObject);
    const feeds = {
        input: new ort.Tensor('float32', inputTensor, [1, 3, TARGET_SIZE[0], TARGET_SIZE[1]])
    };

    const results = await detectionModel.run(feeds);
    const heatmapData = results.output.data;

    const heatmapCanvas = document.createElement('canvas');
    heatmapCanvas.width = TARGET_SIZE[0];
    heatmapCanvas.height = TARGET_SIZE[1];
    const ctx = heatmapCanvas.getContext('2d');
    const imageData = ctx.createImageData(TARGET_SIZE[0], TARGET_SIZE[1]);
    
    for (let i = 0; i < heatmapData.length; i++) {
        imageData.data[i * 4] = heatmapData[i] * 255;     // R
        imageData.data[i * 4 + 1] = heatmapData[i] * 255; // G
        imageData.data[i * 4 + 2] = heatmapData[i] * 255; // B
        imageData.data[i * 4 + 3] = 255;                  // A
    }
    
    ctx.putImageData(imageData, 0, 0);
    return heatmapCanvas;
}

function preprocessImageForRecognition(crops) {
    const targetSize = [32, 128];
    const preprocessedCrops = [];

    for (const crop of crops) {
        const canvas = document.createElement('canvas');
        canvas.width = targetSize[1];
        canvas.height = targetSize[0];
        const ctx = canvas.getContext('2d');
        ctx.drawImage(crop, 0, 0, targetSize[1], targetSize[0]);

        const imageData = ctx.getImageData(0, 0, targetSize[1], targetSize[0]);
        const data = imageData.data;
        const preprocessedData = new Float32Array(targetSize[0] * targetSize[1] * 3);

        for (let i = 0; i < data.length; i += 4) {
            const idx = i / 4;
            preprocessedData[idx] = (data[i] / 255 - REC_MEAN) / REC_STD;       // R
            preprocessedData[idx + targetSize[0] * targetSize[1]] = (data[i + 1] / 255 - REC_MEAN) / REC_STD;  // G
            preprocessedData[idx + targetSize[0] * targetSize[1] * 2] = (data[i + 2] / 255 - REC_MEAN) / REC_STD;  // B
        }

        preprocessedCrops.push(preprocessedData);
    }

    return preprocessedCrops;
}

async function detectAndRecognizeText(imageElement) {
    const heatmapCanvas = await getHeatMapFromImage(imageElement);
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

    const batchSize = isMobile() ? 32 : 8;
    for (let i = 0; i < crops.length; i += batchSize) {
        const batch = crops.slice(i, i + batchSize);
        const preprocessedCrops = preprocessImageForRecognition(batch.map(crop => crop.canvas));

        const recognitionResults = await Promise.all(preprocessedCrops.map(async (cropData) => {
            const feeds = {
                input: new ort.Tensor('float32', cropData, [1, 3, 32, 128])
            };
            const results = await recognitionModel.run(feeds);
            return results.output;
        }));

        recognitionResults.forEach((result, index) => {
            const bestPath = Array.from(result);
            const word = decodeText([new ort.Tensor('float32', bestPath)]);

            if (word && batch[index]) {
                extractedData.push({
                    word: word,
                    boundingBox: batch[index].bbox
                });
            }
        });
    }
    
    return extractedData;
}

