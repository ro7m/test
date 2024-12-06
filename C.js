function softmax(arr) {
    if (!arr || arr.length === 0) return [];

    // Numerical stability improvement
    const max = Math.max(...arr);
    const exp = arr.map(x => Math.exp(x - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    
    return exp.map(x => x / sum);
}

function argmax(arr) {
    if (!arr || arr.length === 0) return -1;

    return arr.reduce((maxIndex, current, index, arr) => 
        current > arr[maxIndex] ? index : maxIndex, 0);
}
function unstackArgmax(probabilities, axis = -1) {
    const shape = probabilities.shape;
    const rank = shape.length;
    
    // Normalize axis if negative
    const normalizedAxis = axis < 0 ? rank + axis : axis;

    // Function to perform argmax along a specific axis
    function argmaxAlongAxis(tensor, axis) {
        // If axis is the last dimension
        if (axis === rank - 1) {
            return tensor.map(row => {
                return row.reduce((maxIndex, current, index, arr) => 
                    current > arr[maxIndex] ? index : maxIndex, 0);
            });
        }
        
        // More complex multi-dimensional argmax would require more elaborate logic
        throw new Error('Argmax for axes other than the last dimension not implemented');
    }

    // Perform argmax
    const bestPath = argmaxAlongAxis(probabilities.data, normalizedAxis);

    return bestPath;
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
