const predictions = await session.run({input: inputTensor});

// Get the output tensor (adjust the output name based on your model)
const outputTensor = predictions.output; // This is likely a Float32Array

// Convert to ORT tensor
const predictionsTensor = new ort.Tensor('float32', outputTensor, predictions.output.dims);

// Perform softmax operation
function softmax(tensor) {
    const data = tensor.data;
    const dims = tensor.dims;
    
    // Softmax along the last dimension
    const result = new Float32Array(data.length);
    
    // Handle multi-dimensional tensor
    const lastDimSize = dims[dims.length - 1];
    const batchSize = data.length / lastDimSize;
    
    for (let i = 0; i < batchSize; i++) {
        const start = i * lastDimSize;
        const end = start + lastDimSize;
        
        // Compute max for numerical stability
        const slice = data.slice(start, end);
        const max = Math.max(...slice);
        
        // Compute exponentials
        const exp = slice.map(x => Math.exp(x - max));
        
        // Compute sum of exponentials
        const sum = exp.reduce((a, b) => a + b, 0);
        
        // Normalize
        for (let j = 0; j < lastDimSize; j++) {
            result[start + j] = exp[j] / sum;
        }
    }
    
    return new ort.Tensor('float32', result, dims);
}

// Perform argmax operation
function argMax(tensor, axis = -1) {
    const data = tensor.data;
    const dims = tensor.dims;
    
    // Determine the size of the axis we're finding max along
    const lastDimSize = dims[dims.length - 1];
    const batchSize = data.length / lastDimSize;
    
    const result = new Int32Array(batchSize);
    
    for (let i = 0; i < batchSize; i++) {
        const start = i * lastDimSize;
        const end = start + lastDimSize;
        
        // Find index of max value in this slice
        const slice = data.slice(start, end);
        const maxIndex = slice.indexOf(Math.max(...slice));
        
        result[i] = maxIndex;
    }
    
    return new ort.Tensor('int32', result, [batchSize]);
}

// Unstack operation
function unstack(tensor, axis = 0) {
    const data = tensor.data;
    const dims = tensor.dims;
    
    // For now, assuming unstacking along the first dimension
    const unstackedTensors = [];
    const unstackedDims = dims.slice(1);
    const elementSize = unstackedDims.reduce((a, b) => a * b, 1);
    
    for (let i = 0; i < dims[0]; i++) {
        const start = i * elementSize;
        const end = start + elementSize;
        const sliceData = data.slice(start, end);
        unstackedTensors.push(new ort.Tensor('int32', sliceData, unstackedDims));
    }
    
    return unstackedTensors;
}

// Modify the unstack function to be compatible with the decode function
function unstackForDecode(tensor) {
    const data = tensor.data;
    const dims = tensor.dims;
    
    // Create a wrapper object that mimics TensorFlow's tensor interface
    const unstackedTensors = [];
    
    // Assuming the tensor is 2D: [batch_size, sequence_length]
    const sequenceLength = dims[1];
    
    for (let i = 0; i < dims[0]; i++) {
        const sequenceData = data.slice(i * sequenceLength, (i + 1) * sequenceLength);
        
        // Create an object that mimics TensorFlow's tensor with dataSync method
        const tensorLike = {
            dataSync: () => sequenceData
        };
        
        unstackedTensors.push(tensorLike);
    }
    
    return unstackedTensors;
}

// Modify the previous workflow
// Apply softmax to predictions
const probabilities = softmax(predictionsTensor);

// Get best path (argmax)
const bestPath = argMax(probabilities, -1);

// Unstack for decoding
const unstackedBestPath = unstackForDecode(bestPath);

// Now use the existing decode function
const words = decodeText(unstackedBestPath);

// Apply softmax to predictions
const probabilities = softmax(predictionsTensor);

// Get best path (argmax)
const bestPath = argMax(probabilities, -1);

// Unstack the best path
const unstackedBestPath = unstack(bestPath, 0);

// Decode text using your existing decodeText function
const words = decodeText(unstackedBestPath[0].data);
