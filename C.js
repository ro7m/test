async function detectText(imageObject, returnModelOutput = false) {
    const inputTensor = preprocessImageForDetection(imageObject);
    const feeds = {
        input: new ort.Tensor('float32', inputTensor, [1, 3, TARGET_SIZE[0], TARGET_SIZE[1]])
    };

    const logits = await detectionModel.run(feeds);
    const logitsData = logits.output.data; // Adjust based on actual model output

    // Sigmoid activation (equivalent to expit in Python)
    const probMap = logitsData.map(val => 1 / (1 + Math.exp(-val)));

    const out = {};

    if (returnModelOutput) {
        out.out_map = probMap;
    }

    // Post-processing step (equivalent to self.postprocessor)
    out.preds = postprocessProbabilityMap(probMap);

    return out;
}

function postprocessProbabilityMap(probMap) {
    // Implement thresholding or other post-processing 
    const threshold = 0.5; // Adjust based on your specific requirements
    return probMap.map(prob => prob > threshold ? 1 : 0);
}

async function detectAndRecognizeText(imageElement) {
    try {
        // Use the new detection method
        const detectionResult = await detectText(imageElement, true);
        
        // Create heatmap from probability map
        const heatmapCanvas = createHeatmapFromProbMap(detectionResult.out_map);
        
        const boundingBoxes = extractBoundingBoxesFromHeatmap(heatmapCanvas, TARGET_SIZE);

        // Rest of the existing recognition logic...
        // Process crops, recognize text, etc.
    } catch (error) {
        console.error('Text detection error:', error);
        throw error;
    }
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
