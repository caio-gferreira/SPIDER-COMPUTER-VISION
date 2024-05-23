document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const formData = new FormData();
    const imageInput = document.getElementById('imageInput');
    formData.append('image', imageInput.files[0]);

    fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log(data)
        const resultDiv = document.getElementById('result');
        resultDiv.innerHTML = `Predicted Class: ${data.predicted_class}`;

        const imageDiv = document.getElementById('image');
        imageDiv.innerHTML = `<img src="data:image/png;base64,${data.image}" alt="Uploaded Image"/>`;
    })
    .catch(error => {
        console.error('Error:', error);
        const resultDiv = document.getElementById('result');
        resultDiv.innerHTML = 'An error occurred while processing the image.';
    });
});