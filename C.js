function softmax(input) {
    // Convert input to an array, handling different types
    const arr = Array.isArray(input) 
        ? input 
        : input && typeof input.data === 'object' 
            ? Array.from(input.data)  // Handle tensor-like objects
            : Array.from(input || []);  // Fallback to empty array if null/undefined
    
    if (!arr || arr.length === 0) return [];

    // Numerical stability improvement
    const max = Math.max(...arr.map(Number));
    const exp = arr.map(x => Math.exp(Number(x) - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    
    return exp.map(x => x / sum);
}

function argmax(arr) {
    if (!arr || arr.length === 0) return -1;

    return arr.reduce((maxIndex, current, index, arr) => 
        current > arr[maxIndex] ? index : maxIndex, 0);
}
function unstackArgmax(probabilities) {
    // If probabilities is a Float32Array, assume it's a flattened 2D array
    // You'll need to provide the original dimensions
    
    // Example implementation assuming a specific shape
    const numRows = probabilities.length / VOCAB.length;  // Assuming VOCAB is defined
    const vocabLength = VOCAB.length;

    // Create an array to store the best path indices
    const bestPath = [];

    // Iterate through each row
    for (let i = 0; i < numRows; i++) {
        // Extract the slice for this row
        const rowStart = i * vocabLength;
        const rowEnd = rowStart + vocabLength;
        const rowProbabilities = probabilities.slice(rowStart, rowEnd);

        // Find the index with maximum probability
        const maxIndex = rowProbabilities.reduce((maxIndex, current, index, arr) => 
            current > arr[maxIndex] ? index : maxIndex, 0);

        bestPath.push(maxIndex);
    }

    return bestPath;
}

// Usage example
async function detectAndRecognizeText(imageElement) {
    const recognitionResults = await recognitionModel.run({ input: inputTensor });
    const probabilities = Object.values(recognitionResults)[0].data;  // Ensure it's converted to data
    
    // You MUST specify the correct number of rows/vocab length
    const bestPath = unstackArgmax(probabilities);
    
    // Convert bestPath indices to characters
    const extractedTexts = bestPath.map(index => VOCAB[index] || '');
}

    

// Example usage in your recognition function
async function detectAndRecognizeText(imageElement) {
    const recognitionResults = await recognitionModel.run({ input: inputTensor });
    const probabilities = Object.values(recognitionResults)[0];
    
    // Equivalent to TensorFlow's tf.unstack(tf.argmax(probabilities, -1), 0)
    const bestPath = unstackArgmax(probabilities);
    
    // Convert bestPath indices to characters
    const extractedTexts = bestPath.map(index => VOCAB[index] || '');
}
