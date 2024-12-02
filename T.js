<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Text Extract</title>
</head>
<body>
    <input type="file" id="imageUpload" accept="image/*">
    <canvas id="canvas" style="display: none;"></canvas>
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/opencv-wasm@4.5.3/build/opencv.js"></script>
    <script>
        async function runDetectionModel(imageData) {
            const session = await ort.InferenceSession.create('detection_model.onnx');
            const image = cv.imdecode(imageData);
            const inputTensor = preprocessImageForDetection(image); // Preprocess the image as required by the detection model
            const feeds = { 'input': inputTensor };
            const results = await session.run(feeds);
            return postprocessDetectionResults(results); // Postprocess the results to get bounding boxes
        }

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

        function preprocessImageForDetection(image) {
            const resizedImage = image.resize(1024, 1024); // Resize to the input size expected by the detection model
            const normalizedImage = resizedImage.float32Array().div(255.0); // Normalize the image
            return new ort.Tensor('float32', normalizedImage, [1, 3, 1024, 1024]); // Convert to ONNX tensor
        }

        function preprocessImageForRecognition(image) {
            const resizedImage = image.resize(128, 32); // Resize to the input size expected by the recognition model
            const normalizedImage = resizedImage.float32Array().div(255.0); // Normalize the image
            return new ort.Tensor('float32', normalizedImage, [1, 3, 32, 128]); // Convert to ONNX tensor
        }

        function postprocessDetectionResults(results) {
            const boundingBoxes = results['output'].data.map(box => ({
                x: box[0],
                y: box[1],
                width: box[2] - box[0],
                height: box[3] - box[1]
            }));
            return boundingBoxes;
        }

        function postprocessRecognitionResult(result) {
            const text = result['output'].data.map(charCode => String.fromCharCode(charCode)).join('');
            return text;
        }

        async function main(file) {
            const reader = new FileReader();
            reader.onload = async (event) => {
                const imageData = new Uint8Array(event.target.result);
                const boundingBoxes = await runDetectionModel(imageData);
                const image = cv.imdecode(imageData);
                const croppedImages = boundingBoxes.map(box => image.getRegion(new cv.Rect(box.x, box.y, box.width, box.height)));
                const recognizedTexts = await runRecognitionModel(croppedImages);
                console.log(recognizedTexts);
            };
            reader.readAsArrayBuffer(file);
        }

        document.getElementById('imageUpload').addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (file) {
                main(file);
            } else {
                console.log('No file selected');
            }
        });

        // Ensure OpenCV.js is loaded before running the script
        function onOpenCvReady() {
            console.log('OpenCV.js is ready');
        }
        if (typeof cv !== 'undefined') {
            onOpenCvReady();
        } else {
            document.addEventListener('load', onOpenCvReady);
        }
    </script>
</body>
</html>
