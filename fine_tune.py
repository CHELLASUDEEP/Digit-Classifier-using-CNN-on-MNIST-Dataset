import os
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd

def preprocess_image(image):
    img = ImageOps.grayscale(image)  # Convert image to grayscale
    img = img.resize((28, 28))  # Resize image to match the input size of the model
    img = np.asarray(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    return img

def retrieve_incorrect_pred(folder_path):
    incorrect_predictions = []
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                # Retrieve incorrect prediction details from the file name
                file_details = file.split("_")
                if len(file_details) >= 5:
                    actual_class = file_details[0]
                    predicted_class = file_details[3]
                    image_name = file_details[4]
                    img = Image.open(file_path)

                    incorrect_predictions.append({
                        "Actual Class": actual_class,
                        "Predicted Class": predicted_class,
                        "Image Name": image_name,
                        "Image": img
                    })
                else:
                    print(f"Invalid filename format: {file}. Skipping this file.")

    return incorrect_predictions

def fine_tune_model(model, incorrect_predictions):
    for example in incorrect_predictions:
        img = preprocess_image(example['Image'])
        actual_label = int(example['Actual Class'])

        actual_label_one_hot = np.zeros((1, 10))
        actual_label_one_hot[0, actual_label] = 1

        model.train_on_batch(img, actual_label_one_hot)

    return model

def load_test_data(csv_path):
    df = pd.read_csv(csv_path)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    test_images = []
    test_labels = []

    for _, row in test_df.iterrows():
        img = row.values[1:].astype(np.float32).reshape(28, 28, 1) / 255.0
        label = to_categorical(row['label'], num_classes=10)

        test_images.append(img)
        test_labels.append(label)

    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    return test_images, test_labels

test_data, test_labels = load_test_data(r'D:\AI ML ASS 1\train.csv')#change the path required to your need

pretrained_model = load_model(r'D:\AI ML ASS 1\model_1.h5')#change the path required to your need

#change the path required to your need
incorrect_pred = retrieve_incorrect_pred(r"C:\Users\CH\Desktop\practice codes\incorrect_predictions")

model_fine = fine_tune_model(pretrained_model, incorrect_pred)

model_fine.save('model_fine.h5')

evaluation = model_fine.evaluate(test_data, test_labels)
print("Test Accuracy:", evaluation[1])