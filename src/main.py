import tensorflow as tf
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import os
app = Flask(__name__)

model = tf.keras.models.load_model('src/v1.keras')

@app.route('/predict', methods=['POST'])
def predict_image():
    file = request.files['image']
    image_data = file.read()

    image = Image.open(io.BytesIO(image_data))
    image = image.resize((224, 224))
    image = np.array(image) / 255.0

    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)

    predicted_class = np.argmax(predictions[0])

    return jsonify({'predicted_class': int(predicted_class)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
