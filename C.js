async function recognizeText(crops, recognitionModel, vocab) {
    // Store original crop information
    const cropInfo = crops.map(crop => ({
        canvas: crop.canvas,
        boundingBox: crop.bbox
    }));

    // Preprocess images (using your existing preprocessing)
    const preprocessedImage = preprocessImageForRecognition(
        cropInfo.map(crop => crop.canvas)
    );

    // Create ONNX Runtime tensor
    const inputTensor = new ort.Tensor('float32', preprocessedImage.data, preprocessedImage.dims);

    // Run inference
    const feeds = { 'input': inputTensor };
    const results = await recognitionModel.run(feeds);

    // Get logits
    const logits = results.logits.data;
    const [batchSize, height, numClasses] = results.logits.dims;

    // Improved decoding function
    function improvedDecoding(logits, vocab) {
        const decodedResults = [];

        for (let b = 0; b < batchSize; b++) {
            let bestText = '';
            let overallConfidence = 1.0;

            // Track the most probable tokens for each position
            for (let h = 0; h < height; h++) {
                const positionLogits = [];
                for (let c = 0; c < numClasses; c++) {
                    const index = b * (height * numClasses) + h * numClasses + c;
                    positionLogits.push({
                        index: c,
                        logit: logits[index]
                    });
                }

                // Sort logits by probability
                const sortedOptions = positionLogits
                    .sort((a, b) => b.logit - a.logit)
                    .filter(option => option.index !== numClasses - 1); // Skip blank/padding token

                // Select top candidate
                if (sortedOptions.length > 0) {
                    const topOption = sortedOptions[0];
                    const character = vocab[topOption.index];
                    
                    // Prevent adding repeated characters
                    if (character && (!bestText.length || character !== bestText[bestText.length - 1])) {
                        bestText += character;
                        // Update confidence (using softmax-like probability)
                        overallConfidence *= Math.exp(topOption.logit) / 
                            positionLogits.reduce((sum, opt) => sum + Math.exp(opt.logit), 0);
                    }
                }
            }

            // Additional filtering and cleaning
            const cleanedText = bestText.trim()
                .replace(/[^a-zA-Z0-9\s]/g, '')  // Remove special characters
                .replace(/\s+/g, ' ');  // Normalize whitespace

            decodedResults.push({
                text: cleanedText,
                confidence: Math.max(0, Math.min(1, overallConfidence)),
                boundingBox: cropInfo[b].boundingBox
            });
        }

        return decodedResults;
    }

    // Perform decoding
    const recognitionResults = improvedDecoding(logits, vocab);

    // Final filtering to remove very short or invalid results
    return recognitionResults.filter(result => 
        result.text.length > 1 && 
        result.confidence > 0.3
    );
}
