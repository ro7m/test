async function recognizeText(crops, recognitionModel, vocab) {
    // Validate inputs
    if (!crops || !recognitionModel || !vocab) {
        throw new Error('Text recognition requires valid inputs');
    }

    // Store original crop information with bounding boxes
    const cropInfo = crops.map(crop => ({
        canvas: crop.canvas,
        boundingBox: crop.boundingBox
    }));

    // Advanced decoding function
    function advancedTextDecoding(logits, vocab) {
        const [batchSize, sequenceLength, vocabSize] = logits.dims;
        const results = [];

        for (let b = 0; b < batchSize; b++) {
            const decodedChars = [];
            let confidenceScore = 1.0;
            let lastCharIndex = -1;

            // Process each position in the sequence
            for (let t = 0; t < sequenceLength; t++) {
                const positionLogits = [];
                
                // Collect logits for this position
                for (let c = 0; c < vocabSize; c++) {
                    const logitIndex = b * (sequenceLength * vocabSize) + t * vocabSize + c;
                    positionLogits.push({
                        index: c,
                        logit: logits.data[logitIndex],
                        char: vocab[c] || ''
                    });
                }

                // Sort and filter logits
                const topCandidates = positionLogits
                    .filter(opt => opt.char !== '')
                    .sort((a, b) => b.logit - a.logit)
                    .slice(0, 3);  // Consider top 3 candidates

                if (topCandidates.length > 0) {
                    const bestCandidate = topCandidates[0];
                    
                    // Prevent excessive repetition
                    if (bestCandidate.index !== lastCharIndex || 
                        (decodedChars.length === 0 || bestCandidate.char !== decodedChars[decodedChars.length - 1])) {
                        decodedChars.push(bestCandidate.char);
                        
                        // Logarithmic confidence calculation
                        confidenceScore *= Math.log1p(Math.exp(bestCandidate.logit)) / 
                                           Math.log1p(Math.exp(positionLogits.reduce((max, p) => Math.max(max, p.logit), -Infinity)));
                    }
                }
            }

            // Text reconstruction and cleaning
            const reconstructedText = decodedChars.join('').trim();

            results.push({
                text: reconstructedText,
                confidence: Math.min(Math.max(confidenceScore, 0), 1),
                rawChars: decodedChars,
                boundingBox: cropInfo[b].boundingBox  // Include original bounding box
            });
        }

        return results;
    }

    // Post-processing function
    function postProcessRecognition(results) {
        return results
            .map(result => {
                // Clean up text
                let cleanedText = result.text
                    .replace(/\s+/g, ' ')  // Normalize whitespace
                    .trim();

                // Additional filtering
                const isValid = cleanedText.length > 1 && 
                                result.confidence > 0.3 && 
                                /[a-zA-Z0-9]/.test(cleanedText);

                return {
                    text: cleanedText,
                    confidence: result.confidence,
                    boundingBox: result.boundingBox,
                    isValid: isValid
                };
            })
            .filter(r => r.isValid)
            .map(({ isValid, ...result }) => result);  // Remove isValid flag from final output
    }

    try {
        // Preprocess images (using your existing preprocessing)
        const preprocessedImage = preprocessImageForRecognition(
            crops.map(crop => crop.canvas)
        );

        // Create ONNX Runtime tensor
        const inputTensor = new ort.Tensor('float32', preprocessedImage.data, preprocessedImage.dims);

        // Run inference
        const feeds = { 'input': inputTensor };
        const results = await recognitionModel.run(feeds);

        // Decode results
        const decodedResults = advancedTextDecoding(results.logits, vocab);

        // Post-process and filter results
        const finalResults = postProcessRecognition(decodedResults);

        return finalResults;

    } catch (error) {
        console.error('Text recognition error:', error);
        return [];
    }
}
