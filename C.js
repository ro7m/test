function softmax(arr) {
    const max = Math.max(...arr);
    const exp = arr.map(x => Math.exp(x - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map(x => x / sum);
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

            // Apply softmax
            const softmaxProbs = softmax(sequenceLogits);

            // Find indices with argmax
            const outIdxs = softmaxProbs.map((val, idx) => ({val, idx}))
                .sort((a, b) => b.val - a.val)
                .map(x => x.idx);

            // Decode indices to characters
            const extractedWord = outIdxs
                .map(idx => VOCAB[idx])
                .join('')
                .split('<eos>')[0]  // If EOS token exists
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
