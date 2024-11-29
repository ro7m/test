const ort = require('onnxruntime-node');
const cv = require('opencv4nodejs');
const fs = require('fs');

// Function to load and run the detection model
async function runDetectionModel(imagePath) {
    const session = await ort.InferenceSession.create('detection_model.onnx');
    const image = cv.imread(imagePath);
    const inputTensor = preprocessImageForDetection(image); // Preprocess the image as required by the detection model
    const feeds = { 'input': inputTensor };
    const results = await session.run(feeds);
    return postprocessDetectionResults(results); // Postprocess the results to get bounding boxes
}

// Function to load and run the recognition model
async function runRecognitionModel(croppedImages) {
    const session = await ort.InferenceSession.create('recognition_model.onnx');
    const results = [];
    for (const croppedImage of croppedImages) {
        const inputTensor = preprocessImageForRecognition(croppedImage); // Preprocess the cropped image as required by the recognition model
        const feeds = { 'input': inputTensor };
        const result = await session.run(feeds);
        results.push(postprocessRecognitionResult(result)); // Postprocess the result to get recognized text
    }
    return results;
}

// Preprocess the image for the detection model (example function, adjust as needed)
function preprocessImageForDetection(image) {
    const resizedImage = image.resize(1024, 1024); // Resize to the input size expected by the detection model
    const normalizedImage = resizedImage.float32Array().div(255.0); // Normalize the image
    return new ort.Tensor('float32', normalizedImage, [1, 3, 1024, 1024]); // Convert to ONNX tensor
}

// Preprocess the image for the recognition model (example function, adjust as needed)
function preprocessImageForRecognition(image) {
    const resizedImage = image.resize(320, 32); // Resize to the input size expected by the recognition model
    const normalizedImage = resizedImage.float32Array().div(255.0); // Normalize the image
    return new ort.Tensor('float32', normalizedImage, [1, 3, 32, 320]); // Convert to ONNX tensor
}

// Postprocess the detection results (example function, adjust as needed)
function postprocessDetectionResults(results) {
    const boundingBoxes = results['output'].data.map(box => ({
        x: box[0],
        y: box[1],
        width: box[2] - box[0],
        height: box[3] - box[1]
    }));
    return boundingBoxes;
}

// Postprocess the recognition result (example function, adjust as needed)
function postprocessRecognitionResult(result) {
    const text = result['output'].data.map(charCode => String.fromCharCode(charCode)).join('');
    return text;
}

async function main() {
    const imagePath = 'path/to/your/image.jpg';
    const boundingBoxes = await runDetectionModel(imagePath);
    const image = cv.imread(imagePath);
    const croppedImages = boundingBoxes.map(box => image.getRegion(new cv.Rect(box.x, box.y, box.width, box.height)));
    const recognizedTexts = await runRecognitionModel(croppedImages);
    console.log(recognizedTexts);
}

main().catch(err => {
    console.error(err);
});
