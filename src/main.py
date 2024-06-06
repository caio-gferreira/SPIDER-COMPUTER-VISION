import base64
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image, ImageDraw
import io

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


app = Flask(__name__)
CORS(app)

detector = tf.keras.models.load_model('./src/v2_1.keras')
class_dict = {
    0: "aranha",
    1: "formiga",
    2: "borboleta",
    3: "mosca",
    4: "tesourinha",
    5: "barata"
}

def process_image(image_data):
    image = Image.open(io.BytesIO(image_data))
    image = image.convert('RGB')
    return image

def detect_object(image):
    image_resized = image.resize((150, 150))
    image_np = np.array(image_resized)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detector(input_tensor)

    return detections

def get_class_predict(detections_np):
    predict_index = np.argmax(detections_np)
    predict_class = class_dict[predict_index]
    predict_score = float(detections_np[0][predict_index])  # Convertendo para float

    return predict_class, predict_score

@app.route('/predict', methods=['POST'])
def predict_image():
    file = request.files['image']
    image_data = file.read()
    image = process_image(image_data)

    detections = detect_object(image)

    predict_class, predict_score = get_class_predict(detections)

    draw = ImageDraw.Draw(image)
    width, height = image.size

    confidence_threshold = 0.5

    if predict_score >= confidence_threshold:
        ymin, xmin, ymax, xmax = 0, 0, 1, 1
        xmin = int(xmin * width)
        ymin = int(ymin * height)
        xmax = int(xmax * width)
        ymax = int(ymax * height)

        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red")

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    truncated_score = "{:.2f}".format(predict_score)

    return jsonify({
        'image': img_str,
        'specie': predict_class,
        'score': truncated_score,
    })





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)