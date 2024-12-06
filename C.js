function softmax(arr) {
    // Prevent NaN by handling edge cases
    if (!arr || arr.length === 0) return [];

    const max = Math.max(...arr);
    const exp = arr.map(x => {
        // Ensure x is a number
        const val = Number(x);
        return isNaN(val) ? 0 : Math.exp(val - max);
    });
    const sum = exp.reduce((a, b) => a + b, 0);
    
    return exp.map(x => sum > 0 ? x / sum : 0);
}

async function detectAndRecognizeText(imageElement) {
    const batchSize = 32;
    let fullText = '';
    
    for (let i = 0; i < crops.length; i += batchSize) {
        const batch = crops.slice(i, i + batchSize);
        const inputTensor = preprocessImageForRecognition(batch.map(crop => crop.canvas));

        const recognitionResults = await recognitionModel.run({ input: inputTensor });
        const logits = Object.values(recognitionResults)[0].data;
        const vocabLength = VOCAB.length;

        const batchTexts = [];
        for (let j = 0; j < batch.length; j++) {
            // Extract sequence logits
            const sequenceLogits = logits.slice(
                j * vocabLength, 
                (j + 1) * vocabLength
            );

            // Debug logging
            console.log('Sequence Logits:', sequenceLogits);
            console.log('Sequence Logits Type:', typeof sequenceLogits[0]);

            // Robust softmax application
            const softmaxProbs = softmax(sequenceLogits);

            // Debug logging
            console.log('Softmax Probs:', softmaxProbs);

            // Find indices with highest probabilities
            const outIdxs = softmaxProbs
                .map((val, idx) => ({val, idx}))
                .sort((a, b) => b.val - a.val)
                .map(x => x.idx);

            // Decode indices to characters
            const extractedWord = outIdxs
                .map(idx => VOCAB[idx] || '')
                .join('')
                .trim();

            batchTexts.push(extractedWord);

            extractedData.push({
                word: extractedWord,
                boundingBox: batch[j].bbox
            });
        }
        
        fullText += batchTexts.join(' ') + ' ';
    }          

    return extractedData;
}
