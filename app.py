import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import os

# Load your trained model
model = tf.keras.models.load_model(r'C:\Users\CH\Desktop\practice codes\model_fine.h5')  # Replace with your model path

# Function to preprocess the image
def preprocess_image(image):
    img = ImageOps.grayscale(image)  # Convert image to grayscale
    img = img.resize((28, 28))  # Resize image to match the input size of the model
    img = np.asarray(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    return img

def predict_image(image):
    processed_img = preprocess_image(image)  # Preprocess the image
    prediction = model.predict(processed_img)
    return np.argmax(prediction)

# Streamlit app
st.title('Image Classification with CNN')

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Predict the image
    predicted_class = predict_image(img)
    st.write(f"Predicted Class: {predicted_class}")

    # Buttons for user feedback
    feedback = st.radio("Was the prediction correct?", ("Correct", "Incorrect"))
    if feedback == "Incorrect":
        actual_class = st.text_input("Enter the actual class (numeric label):")
        if actual_class.strip() != "":
            try:
                actual_class_int = int(actual_class)
                # Create 'incorrect_predictions' directory if it doesn't exist
                if not os.path.exists("incorrect_predictions"):
                    os.makedirs("incorrect_predictions")

                # Save details for incorrect feedback
                file_name = f"incorrect_predictions/{actual_class_int}_predicted_as_{predicted_class}_{uploaded_file.name}"
                img.save(file_name)
                st.write(f"Incorrect feedback saved at: {file_name}")
            except ValueError:
                st.warning("Please enter a valid numeric label for the actual class.")