import numpy as np
import pickle
from mtcnn import MTCNN
import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import VGG16
import os
import cv2
import pathlib
import tensorflow as tf

# Add necessary imports
from tqdm import tqdm

# Define the directory for saving uploaded images
UPLOADS_DIR = 'uploads'
pathlib.Path(UPLOADS_DIR).mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

detector = MTCNN(min_face_size=20, scale_factor=0.7, steps_threshold=[0.6, 0.7, 0.7])

vgg16_weights_path = "/Users/jasmeet/Downloads/vgg16_weights_tf_dim_ordering_tf_kernels (1).h5"
# Initialize VGG16 model
model = VGG16(
    include_top=True,
    weights=vgg16_weights_path,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax")

# Load feature_list and filenames
feature_list = np.array(pickle.load(open('embedding.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))


def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join(UPLOADS_DIR, uploaded_image.name), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False


def extract_features(img_path, model, detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)

    if results:
        x, y, width, height = results[0]['box']
        face = img[y:y + height, x:x + width]

        image = Image.fromarray(face)
        image = image.resize((224, 224))
        face_array = np.asarray(image)
        face_array = face_array.astype('float32')

        expanded_img = np.expand_dims(face_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img)
        return preprocessed_img
    else:
        return None


def calculate_similarity(feature, feature_list):
    # Compute cosine similarity with all features
    similarity = tf.map_fn(lambda x: tf.reduce_sum(tf.multiply(feature, x)) / (tf.norm(feature) * tf.norm(x)),
                           feature_list)
    return similarity


def recommend(feature_list, features):
    similarity = calculate_similarity(features, feature_list)
    index_pos = tf.argmax(similarity).numpy()
    return index_pos


st.title('Which Bollywood celebrity are you?')

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Save the image in the directory
    if save_uploaded_image(uploaded_image):
        # Load the image
        display_image = Image.open(uploaded_image)
        img_path = os.path.join(UPLOADS_DIR, uploaded_image.name)
        preprocessed_img = extract_features(img_path, model, detector)

        if preprocessed_img is not None:
            # Extract features using VGG16
            features = model.predict(preprocessed_img).flatten()

            # Recommend
            index_pos = recommend(feature_list, features)
            if 0 <= index_pos < len(filenames):
                predicted_actor = " ".join(filenames[index_pos].split('/')[-1].split('_'))
            else:
                predicted_actor = "Unknown"

            # Display the results
            col1, col2 = st.columns(2)
            with col1:
                st.header('Your uploaded image')
                st.image(display_image)
            with col2:
                st.header("Seems like " + predicted_actor)
                st.image(filenames[index_pos], width=300)
        else:
            st.error("No face detected in the uploaded image.")
    else:
        st.error("Failed to save the uploaded image.")
