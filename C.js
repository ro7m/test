function preprocessImageForRecognition(crops, targetSize = [32, 128], mean = [0.694, 0.695, 0.693], std = [0.299, 0.296, 0.301]) {
    // Helper function to resize and pad image
    function resizeAndPadImage(imageData) {
        // Create canvas for resizing
        const canvas = new OffscreenCanvas(imageData.width, imageData.height);
        const ctx = canvas.getContext('2d');
        ctx.drawImage(imageData, 0, 0);

        const [targetHeight, targetWidth] = targetSize;
        let resizedWidth, resizedHeight;
        let aspectRatio = targetWidth / targetHeight;

        // Calculate resize dimensions maintaining aspect ratio
        if (aspectRatio * imageData.height > imageData.width) {
            resizedHeight = targetHeight;
            resizedWidth = Math.round((targetHeight * imageData.width) / imageData.height);
        } else {
            resizedWidth = targetWidth;
            resizedHeight = Math.round((targetWidth * imageData.height) / imageData.width);
        }

        // Create resize canvas
        const resizeCanvas = new OffscreenCanvas(targetWidth, targetHeight);
        const resizeCtx = resizeCanvas.getContext('2d');
        
        // Fill with black background
        resizeCtx.fillStyle = 'black';
        resizeCtx.fillRect(0, 0, targetWidth, targetHeight);

        // Calculate positioning for resize
        const xOffset = Math.floor((targetWidth - resizedWidth) / 2);
        const yOffset = Math.floor((targetHeight - resizedHeight) / 2);

        // Draw resized image
        resizeCtx.drawImage(
            canvas, 
            0, 0, imageData.width, imageData.height, 
            xOffset, yOffset, resizedWidth, resizedHeight
        );

        // Get image data
        const imageDataResized = resizeCtx.getImageData(0, 0, targetWidth, targetHeight);
        return imageDataResized;
    }

    // Process each crop
    const processedImages = crops.map(crop => {
        // Resize and pad the image
        const resizedImage = resizeAndPadImage(crop);
        
        // Allocate a new Float32Array for the entire image (3 channels)
        const float32Data = new Float32Array(3 * targetSize[0] * targetSize[1]);
        
        // Normalize and separate channels
        for (let y = 0; y < targetSize[0]; y++) {
            for (let x = 0; x < targetSize[1]; x++) {
                const pixelIndex = (y * targetSize[1] + x) * 4;
                const channelSize = targetSize[0] * targetSize[1];
                
                // Extract RGB and normalize
                const r = resizedImage.data[pixelIndex] / 255.0;
                const g = resizedImage.data[pixelIndex + 1] / 255.0;
                const b = resizedImage.data[pixelIndex + 2] / 255.0;

                // Separate channels with normalization
                float32Data[y * targetSize[1] + x] = (r - mean[0]) / std[0];  // R channel
                float32Data[channelSize + y * targetSize[1] + x] = (g - mean[1]) / std[1];  // G channel
                float32Data[2 * channelSize + y * targetSize[1] + x] = (b - mean[2]) / std[2];  // B channel
            }
        }

        return float32Data;
    });

    // Concatenate multiple processed images
    if (processedImages.length > 1) {
        const combinedLength = processedImages[0].length * processedImages.length;
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


const imageCrops = [imageData1, imageData2];
const preprocessedImage = preprocessImageForRecognition(imageCrops);

// Create ONNX Runtime tensor
const tensor = new ort.Tensor('float32', preprocessedImage.data, preprocessedImage.dims);


----

    async function recognizeText(crops, recognitionModel, vocab) {
    // Preprocess images
    const preprocessedImage = preprocessImageForRecognition(
        crops.map(crop => crop.canvas)
    );

    // Create ONNX Runtime tensor
    const inputTensor = new ort.Tensor('float32', preprocessedImage.data, preprocessedImage.dims);

    // Run inference
    const feeds = { 'input': inputTensor };
    const results = await recognitionModel.run(feeds);

    // Get logits (assuming the output is named 'output' or 'logits')
    const logits = results.logits.data;

    // Softmax implementation
    function softmax(arr, axis = -1) {
        // Find max value for numerical stability
        const maxVal = Math.max(...arr);
        
        // Exponentiate and normalize
        const exp = arr.map(x => Math.exp(x - maxVal));
        const sumExp = exp.reduce((a, b) => a + b, 0);
        
        return exp.map(x => x / sumExp);
    }

    // Apply softmax to logits
    // In ONNX Runtime, we'll need to do this manually across the last axis
    const probabilities = [];
    const [batchSize, classes, sequenceLength] = preprocessedImage.dims;

    // Softmax for each sequence in the batch
    for (let b = 0; b < batchSize; b++) {
        const batchProbs = [];
        for (let t = 0; t < sequenceLength; t++) {
            // Extract logits for this timestep
            const timestepLogits = [];
            for (let c = 0; c < classes; c++) {
                const index = b * (classes * sequenceLength) + c * sequenceLength + t;
                timestepLogits.push(logits[index]);
            }
            
            // Apply softmax to timestep
            batchProbs.push(softmax(timestepLogits));
        }
        probabilities.push(batchProbs);
    }

    // Find argmax (best path) similar to tf.argMax
    const bestPath = probabilities.map(batchProb => 
        batchProb.map(timestep => 
            timestep.indexOf(Math.max(...timestep))
        )
    );

    // Convert best path to text using vocab
    const decodedTexts = bestPath.map(sequence => 
        sequence
            .filter(idx => idx !== vocab.length)  // Remove blank token
            .map(idx => vocab[idx])
            .join('')
    );

    return {
        probabilities,
        bestPath,
        decodedTexts
    };
}

// Example usage
async function processRecognition(crops, recognitionModel, vocab) {
    try {
        const results = await recognizeText(crops, recognitionModel, vocab);
        console.log('Decoded Texts:', results.decodedTexts);
        console.log('Best Path Indices:', results.bestPath);
    } catch (error) {
        console.error('Recognition error:', error);
    }
}

----
// Assuming you have:
// - crops: array of image crops
// - recognitionModel: ONNX Runtime session
// - vocab: array of characters/tokens
const results = await recognizeText(crops, recognitionModel, vocab);
console.log(results.decodedTexts);
    
    
