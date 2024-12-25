import streamlit as st 
import h5py
import tensorflow as tf
import cv2 as cv
import numpy as np
from PIL import Image


def load_model_from_hdf5(file_path):
    with h5py.File(file_path, 'r') as f:
        model = tf.keras.models.load_model(f)
    return model


def preprocess_image(image_path, target_size=(300, 300)):

    image = np.array(image_path)
    # Resize image to target size
    image = cv.resize(image, target_size)
    # Normalize pixel values to [0, 1]
    image = image / 255.0
    #Expand dimentions
    image = np.expand_dims(image, axis=0)
    return image

class_labels = {"Not good quality":0,"Good Quality":1}

st.title("Lemon Quality APP")
st.header("Upload lemon photo and check if it is good quality or not ")

#uploading the image 
uploaded_file = st.file_uploader("upload the car image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded file as a PIL image
    image_input = Image.open(uploaded_file)
    
    # Display the image using Streamlit
    st.image(image_input, caption="Uploaded Image", use_column_width=True)



if st.button("Check Lemons Quality"):
    image = preprocess_image(image_input)  # Now image is defined
    model = load_model_from_hdf5("F:\\github\\streamlit\\lemon_quality\\model (5).hdf5")
    prediction = model.predict(image)
    predicted_class_label = np.argmax(prediction)
    lemon_Quality = list(class_labels.keys())[predicted_class_label]
    st.write("The lemon is in ", lemon_Quality)