from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from extract_bottleneck_features import *
from tensorflow.keras.models import load_model
from sklearn.datasets import load_files
from tensorflow.keras import utils
from PIL import ImageFile
from extract_bottleneck_features import *
import cv2
import matplotlib.pyplot as plt
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True
ResNet50_model = ResNet50(weights='imagenet')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        # Perform image analysis
        result = analyze_image(filename)
        
        return render_template('index.html', uploaded_image=filename, result=result)
    else:
        return "No file selected"

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

def Resnet50_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = ResNet50_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]

def analyze_image(image_path):
    # Preprocess the image
    tensor = path_to_tensor(image_path)

    if dog_detector(image_path):
        breed = Resnet50_predict_breed(image_path)
        return f"A dog is detected! It looks like a {breed}."

    elif face_detector(image_path):
        resembling_breed = Resnet50_predict_breed(image_path)
        return f"A human is detected! You look like a {resembling_breed}."

    else:
        return "Neither a dog nor a human is detected in the image."

if __name__ == '__main__':
    app.run(debug=True, port=5500)  # Change port to 8080 or any other available port
