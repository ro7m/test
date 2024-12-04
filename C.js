// Constants (keep existing constants)
const LOADING_STAGES = {
    INITIALIZING: 'Initializing application...',
    LOADING_MODELS: 'Loading detection and recognition models...',
    SETTING_UP_CAMERA: 'Setting up camera...',
    READY: 'Ready to capture',
};

// Modify existing DOM element selections and add loading message element
const loadingMessageElement = document.getElementById('loadingMessage');

// Performance tracking function
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

// Update loading stages function
function updateLoadingStage(stage) {
    console.log(`Stage: ${stage}`);
    if (loadingMessageElement) {
        loadingMessageElement.textContent = stage;
    }
}

// Improved model loading with detailed progress
async function loadModels() {
    updateLoadingStage(LOADING_STAGES.LOADING_MODELS);
    performanceTracker.start('Model Loading');
    
    try {
        detectionModel = await ort.InferenceSession.create('models/rep_fast_base.onnx', {
            executionProviders: ['webgl', 'wasm'] // Add multiple providers for better performance
        });
        recognitionModel = await ort.InferenceSession.create('models/parseq_dynamic.onnx', {
            executionProviders: ['webgl', 'wasm']
        });
        
        const modelLoadTime = performanceTracker.end('Model Loading');
        console.log(`Models loaded successfully in ${modelLoadTime.toFixed(2)}ms`);
    } catch (error) {
        console.error('Error loading models:', error);
        updateLoadingStage('Failed to load models. Please refresh.');
    }
}

// Improved camera setup with progress tracking
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

// Improved bounding box extraction with advanced contour detection
function extractBoundingBoxes(probMap, imageElement) {
    // More sophisticated bounding box extraction
    const threshold = 0.7; // Increased threshold for more confident detections
    const boxes = [];
    const width = 1024;
    const height = 1024;
    const scaleX = imageElement.width / width;
    const scaleY = imageElement.height / height;

    // Create a 2D array to track visited pixels
    const visited = new Array(height).fill(null).map(() => new Array(width).fill(false));
    
    function expandBox(x, y) {
        let minX = x, maxX = x, minY = y, maxY = y;
        const queue = [[x, y]];
        
        while (queue.length > 0) {
            const [currX, currY] = queue.pop();
            
            // Check neighboring pixels
            const directions = [
                [0, 1], [0, -1], [1, 0], [-1, 0],
                [1, 1], [1, -1], [-1, 1], [-1, -1]
            ];
            
            for (const [dx, dy] of directions) {
                const newX = currX + dx;
                const newY = currY + dy;
                
                if (newX >= 0 && newX < width && newY >= 0 && newY < height &&
                    !visited[newY][newX] && probMap[newY * width + newX] > threshold) {
                    
                    visited[newY][newX] = true;
                    queue.push([newX, newY]);
                    
                    minX = Math.min(minX, newX);
                    maxX = Math.max(maxX, newX);
                    minY = Math.min(minY, newY);
                    maxY = Math.max(maxY, newY);
                }
            }
        }
        
        return [minX, minY, maxX, maxY];
    }
    
    // Scan through the probability map
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const index = y * width + x;
            if (!visited[y][x] && probMap[index] > threshold) {
                visited[y][x] = true;
                const box = expandBox(x, y);
                
                // Filter out very small boxes
                const boxWidth = box[2] - box[0];
                const boxHeight = box[3] - box[1];
                if (boxWidth > 10 && boxHeight > 10) {
                    boxes.push([
                        box[0] * scaleX,
                        box[1] * scaleY,
                        box[2] * scaleX,
                        box[3] * scaleY
                    ]);
                }
            }
        }
    }
    
    return boxes;
}

// Modified text detection and recognition with performance tracking
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

// Modify existing init function to include detailed tracking
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
