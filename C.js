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

async function getHeatMapFromImage(imageObject) {
    const inputTensor = preprocessImageForDetection(imageObject);
    const feeds = {
        input: new ort.Tensor('float32', inputTensor, [1, 3, TARGET_SIZE[0], TARGET_SIZE[1]])
    };

    const results = await detectionModel.run(feeds);
    const heatmapData = results.logits;

    const heatmapCanvas = document.createElement('canvas');
    heatmapCanvas.width = TARGET_SIZE[0];
    heatmapCanvas.height = TARGET_SIZE[1];
    const ctx = heatmapCanvas.getContext('2d');
    const imageData = ctx.createImageData(TARGET_SIZE[0], TARGET_SIZE[1]);
    
    // Normalize heatmap data
    const maxValue = Math.max(...heatmapData);
    const minValue = Math.min(...heatmapData);
    
    for (let i = 0; i < heatmapData.length; i++) {
        // Normalize and enhance contrast
        const normalizedValue = (heatmapData[i] - minValue) / (maxValue - minValue);
        const thresholdedValue = normalizedValue > 0.5 ? 255 : 0;  // Binary thresholding
        
        imageData.data[i * 4] = thresholdedValue;     // R
        imageData.data[i * 4 + 1] = thresholdedValue; // G
        imageData.data[i * 4 + 2] = thresholdedValue; // B
        imageData.data[i * 4 + 3] = 255;              // A
    }
    
    ctx.putImageData(imageData, 0, 0);
    return heatmapCanvas;
}

function extractBoundingBoxesFromHeatmap(heatmapCanvas, size) {
    let src = cv.imread(heatmapCanvas);
    cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
    
    // Apply adaptive thresholding for better text region detection
    cv.adaptiveThreshold(src, src, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2);
    
    // Morphological operations to clean up and connect text regions
    let kernel = cv.Mat.ones(3, 3, cv.CV_8U);
    cv.morphologyEx(src, src, cv.MORPH_CLOSE, kernel);
    cv.morphologyEx(src, src, cv.MORPH_OPEN, kernel);
    
    // Find contours with more robust filtering
    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat();
    cv.findContours(src, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    
    const boundingBoxes = [];
    const minArea = 50;  // Minimum contour area to consider
    const maxArea = size[0] * size[1] * 0.3;  // Maximum area (30% of image)
    
    for (let i = 0; i < contours.size(); ++i) {
        const contour = contours.get(i);
        const area = cv.contourArea(contour);
        
        // Filter contours based on area and aspect ratio
        if (area > minArea && area < maxArea) {
            const contourBoundingBox = cv.boundingRect(contour);
            const aspectRatio = contourBoundingBox.width / contourBoundingBox.height;
            
            // Only add contours with a reasonable aspect ratio (not too wide or tall)
            if (aspectRatio > 0.1 && aspectRatio < 10) {
                boundingBoxes.push(transformBoundingBox(contourBoundingBox, i, size));
            }
        }
        
        // Clean up to avoid memory leaks
        contour.delete();
    }
    
    // Clean up OpenCV resources
    src.delete();
    contours.delete();
    hierarchy.delete();
    kernel.delete();
    
    return boundingBoxes;
    }
