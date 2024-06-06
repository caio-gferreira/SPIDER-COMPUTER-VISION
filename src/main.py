import base64
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image, ImageDraw
import io

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model('src/v1.keras')

def process_image(image_data):
    image = Image.open(io.BytesIO(image_data))
    
    image = image.resize((150, 150))
    image = image.convert('RGB')

    image_np = np.array(image) / 255.0
    image_np = np.expand_dims(image_np, axis=0)

    return image, image_np

@app.route('/predict', methods=['POST'])
def predict_image():
    file = request.files['image']
    image_data = file.read()
    image, image_np = process_image(image_data)

    predictions = model.predict(image_np)
    predicted_class = predictions[0][0]

    draw = ImageDraw.Draw(image)
    width, height = image.size
    draw.rectangle(((0, 0), (width, height)), outline="red", width=5)

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return jsonify({
        'predicted_class': format(predicted_class, '.2f'),
        'image': img_str 
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
