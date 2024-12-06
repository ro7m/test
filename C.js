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
