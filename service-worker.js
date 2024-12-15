const CACHE_NAME = 'text-extract-v1';
const urlsToCache = [
  '/Onnx-models/',  // Add base path
  '/Onnx-models/models/',
  '/Onnx-models/index.html',
  '/Onnx-models/styles.css',
  '/Onnx-models/app.js',
  'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(urlsToCache))
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => response || fetch(event.request))
  );
});

if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/Onnx-models/service-worker.js', {
      scope: '/Onnx-models/'
    }).then(registration => {
      console.log('ServiceWorker registration successful');
    }).catch(err => {
      console.log('ServiceWorker registration failed: ', err);
    });
  });
}