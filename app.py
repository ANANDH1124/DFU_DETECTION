import streamlit as st
import numpy as np
import PIL.Image as Image
import tensorflow as tf
from tensorflow import keras

# Load the trained model
model = keras.models.load_model("model.h5")

# Define the class labels
class_labels = {
    0: 'Abnormal(Ulcer)',
    1: 'Normal(Healthy skin)'
}

# Function to make predictions
def predict(image):
    # Preprocess the image
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    image = image[np.newaxis, ...]

    # Make a prediction
    result = model.predict(image)
    
    # Get the predicted class label
    predicted_class = np.argmax(result)

    return class_labels[predicted_class]

# Streamlit app
st.title("dfu Classifier")

# File uploader
uploaded_image = st.file_uploader("Upload a dfu image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make a prediction when the user clicks the button
    if st.button("Classify"):
        prediction = predict(image)
        st.write(f"Prediction: {prediction}")
