import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from io import BytesIO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cnn_model.h5")

# Load model once
cnn_model = load_model(MODEL_PATH)

def preprocess_image(file_storage, target_size=(224, 224)):
    """
    Preprocess Flask uploaded image for CNN
    """
    # Convert FileStorage -> BytesIO
    image_bytes = BytesIO(file_storage.read())
    
    # Load image using Keras
    img = image.load_img(image_bytes, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def predict_image(img_array):
    prediction = cnn_model.predict(img_array)[0][0]
    return float(prediction)
