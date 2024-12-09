async function recognizeText(crops, recognitionModel, vocab) {
    // Validate inputs
    if (!crops || !recognitionModel || !vocab) {
        throw new Error('Text recognition requires valid inputs');
    }

    // Advanced decoding function tailored to [17, 32, 127] logits
    function advancedTextDecoding(logits, vocab) {
        const [batchSize, sequenceLength, vocabSize] = logits.dims;
        const results = [];

        for (let b = 0; b < batchSize; b++) {
            const decodedChars = [];
            let confidenceScore = 1.0;

            // Process each position in the sequence
            for (let t = 0; t < sequenceLength; t++) {
                const positionLogits = [];
                
                // Collect logits for this position
                for (let c = 0; c < vocabSize; c++) {
                    const logitIndex = b * (sequenceLength * vocabSize) + t * vocabSize + c;
                    positionLogits.push({
                        index: c,
                        logit: logits.data[logitIndex],
                        char: c < vocab.length ? vocab[c] : ''
                    });
                }

                // More aggressive candidate selection
                const topCandidates = positionLogits
                    .filter(opt => opt.char && opt.char !== '')
                    .sort((a, b) => b.logit - a.logit)
                    .slice(0, 3);

                if (topCandidates.length > 0) {
                    const bestCandidate = topCandidates[0];
                    
                    // Intelligent character addition
                    if (decodedChars.length === 0 || 
                        bestCandidate.char !== decodedChars[decodedChars.length - 1]) {
                        decodedChars.push(bestCandidate.char);
                        
                        // Confidence calculation
                        const softmaxDenom = positionLogits.reduce((sum, p) => 
                            sum + Math.exp(p.logit), 0);
                        
                        confidenceScore *= Math.exp(bestCandidate.logit) / softmaxDenom;
                    }
                }
            }

            // Reconstruct text
            const reconstructedText = decodedChars.join('').trim();

            results.push({
                text: reconstructedText,
                confidence: Math.min(Math.max(confidenceScore, 0), 1),
                rawChars: decodedChars
            });
        }

        return results;
    }

    // Post-processing function with additional filtering
    function postProcessRecognition(results, crops) {
        return results
            .map((result, index) => {
                // Clean up text
                let cleanedText = result.text
                    .replace(/\s+/g, ' ')  // Normalize whitespace
                    .trim();

                // Additional filtering
                const isValid = cleanedText.length > 1 && 
                                result.confidence > 0.2 && 
                                /[a-zA-Z0-9]/.test(cleanedText);

                return {
                    text: cleanedText,
                    confidence: result.confidence,
                    boundingBox: crops[index].boundingBox,
                    isValid: isValid
                };
            })
            .filter(r => r.isValid)
            .map(({ isValid, ...result }) => result);  // Remove isValid flag
    }

    try {
        // Preprocess images 
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
        const finalResults = postProcessRecognition(decodedResults, crops);

        // Debugging output
        console.log('Recognition Results:', finalResults);

        return finalResults;

    } catch (error) {
        console.error('Text recognition comprehensive error:', error);
        // Log full error details
        console.error('Error Details:', {
            message: error.message,
            stack: error.stack
        });
        return [];
    }
}
