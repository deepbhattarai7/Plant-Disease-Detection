import os
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Set the upload directory path
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your model with the correct path
model_path = os.path.join(os.path.dirname(__file__), '../model/model.h5')
model = load_model(model_path)

class_names = ['Healthy', 'Early Blight', 'Late Blight']

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        img_array = prepare_image(file_path)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction))
        result = {'class': class_names[predicted_class], 'confidence': confidence}
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
