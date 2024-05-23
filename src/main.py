import base64
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model('src/v1.keras')

@app.route('/predict', methods=['POST'])
def predict_image():
    file = request.files['image']
    image_data = file.read()

    image = Image.open(io.BytesIO(image_data))
    
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    image = image.resize((28, 28))
    image = image.convert('L')
    image_np = np.array(image) / 255.0

    image_np = np.expand_dims(image_np, axis=0)
    image_np = np.expand_dims(image_np, axis=-1) 

    predictions = model.predict(image_np)
    print(predictions)
    predicted_class = np.argmax(predictions[0])

    return jsonify({
        'predicted_class': float(predicted_class),
        'mask': mask_str,
        'image': img_str
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
