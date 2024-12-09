async function recognizeText(crops, recognitionModel, vocab) {
    // Store original crop information
    const cropInfo = crops.map(crop => ({
        canvas: crop.canvas,
        boundingBox: crop.boundingBox // Assuming original crops had bounding box info
    }));

    // Preprocess images
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

    // Advanced softmax with beam search
    function beamSearchDecoding(logits, vocab, beamWidth = 3) {
        const decodedResults = [];

        for (let b = 0; b < batchSize; b++) {
            // Extract logits for this batch
            const batchLogits = [];
            for (let h = 0; h < height; h++) {
                const positionLogits = [];
                for (let c = 0; c < numClasses; c++) {
                    const index = b * (height * numClasses) + h * numClasses + c;
                    positionLogits.push({
                        index: c,
                        logit: logits[index]
                    });
                }
                
                // Sort logits by probability (descending)
                const sortedLogits = positionLogits
                    .sort((a, b) => b.logit - a.logit)
                    .slice(0, beamWidth);
                
                batchLogits.push(sortedLogits);
            }

            // Combine top candidates
            const candidateTexts = generateCandidateTexts(batchLogits, vocab);
            
            // Select best candidate
            const bestCandidate = candidateTexts.reduce((best, current) => 
                (!best || current.confidence > best.confidence) ? current : best
            );

            decodedResults.push({
                text: bestCandidate.text,
                confidence: bestCandidate.confidence,
                boundingBox: cropInfo[b].boundingBox
            });
        }

        return decodedResults;
    }

    // Helper function to generate candidate texts
    function generateCandidateTexts(batchLogits, vocab) {
        const candidates = [{ 
            text: '', 
            confidence: 1, 
            path: [] 
        }];

        for (const positionOptions of batchLogits) {
            const newCandidates = [];

            for (const candidate of candidates) {
                for (const option of positionOptions) {
                    // Skip blank token (usually last index)
                    if (option.index === numClasses - 1) continue;

                    const newText = candidate.text + vocab[option.index];
                    const newConfidence = candidate.confidence * Math.exp(option.logit);

                    newCandidates.push({
                        text: newText,
                        confidence: newConfidence,
                        path: [...candidate.path, option.index]
                    });
                }
            }

            // Keep top candidates
            candidates.length = 0;
            candidates.push(...newCandidates
                .sort((a, b) => b.confidence - a.confidence)
                .slice(0, 3)
            );
        }

        return candidates;
    }

    // Perform decoding
    const recognitionResults = beamSearchDecoding(logits, vocab);

    // Post-processing to clean up and improve text
    const processedResults = recognitionResults.map(result => ({
        text: result.text.trim(),
        confidence: result.confidence,
        boundingBox: result.boundingBox
    }));

    return processedResults;
}
