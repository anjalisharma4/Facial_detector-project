# load img -> face detection and extract its features
# find the cosine distance of current image with all the 8656 features
# recommend that image

import os
import cv2
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import VGG16
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

# Load filenames and feature_list
filenames = pickle.load(open("filenames.pkl", "rb"))
feature_list = np.array(pickle.load(open('embedding.pkl', 'rb')))

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

# Initialize MTCNN detector
detector = MTCNN(min_face_size=20, scale_factor=0.7, steps_threshold=[0.6, 0.7, 0.7])

@tf.function
def extract_features(images):
    # Preprocess input for VGG16 model
    preprocessed_imgs = preprocess_input(images)
    # Extract features using VGG16
    results = model(preprocessed_imgs)
    return results

@tf.function
def calculate_similarity(feature, feature_list):
    # Compute cosine similarity with all features
    similarity = tf.map_fn(lambda x: tf.reduce_sum(tf.multiply(feature, x)) / (tf.norm(feature) * tf.norm(x)), feature_list)
    return similarity

# Load sample image
sample_img_path = "/Users/jasmeet/PycharmProjects/pythonProject1/pythonProject2 2/sample/sample/FB_IMG_1483008528049.jpg"
sample_img = cv2.imread(sample_img_path)

# Detect faces in the sample image
results = detector.detect_faces(sample_img)
#print(detector.detect_faces(sample_img))
if results:
    x, y, width, height = results[0]['box']
    face = sample_img[y:y + height, x:x + width]

    # Resize face image to 224x224
    face_image = cv2.resize(face, (224, 224))

    # Add batch dimension
    images_batch = np.expand_dims(face_image, axis=0)

    # Extract features using VGG16
    features = extract_features(images_batch)

    # Calculate cosine similarity
    similarity = calculate_similarity(features, feature_list)

    # Find index of most similar image
    index_pos = tf.argmax(similarity).numpy()

    print("Calculated Index Position:", index_pos)
    print("Index position:", index_pos)
    print("Number of filenames:", len(filenames))

    # Check if index_pos is within valid range
    if 0 <= index_pos < len(filenames):
        temp_img_path = filenames[index_pos]
        print("Filename at Index Position:", temp_img_path)
        # Check if the file exists before reading it
        if os.path.exists(temp_img_path):
            temp_img = cv2.imread(temp_img_path)
            cv2.imshow('output', temp_img)
            cv2.waitKey(0)
        else:
            print(f"File '{temp_img_path}' does not exist.")
    else:
        print("Index position is out of range.")
else:
    print("No face detected in the sample image.")
