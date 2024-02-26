from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import pickle
from tqdm import tqdm
import os
from PIL import Image


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
data_dir = "/Users/jasmeet/PycharmProjects/pythonProject1/pythonProject2 2/data"


filenames = []
for actor in os.listdir(data_dir):
    actor_dir = os.path.join(data_dir, actor)
    if os.path.isdir(actor_dir):
        for file in os.listdir(actor_dir):
            # Check if the file has a valid image extension
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                filenames.append(os.path.join(actor_dir, file))

# Save filenames to filenames.pkl
with open('filenames.pkl', 'wb') as f:
    pickle.dump(filenames, f)

# Load filenames from filenames.pkl
with open('filenames.pkl', 'rb') as f:
    filenames = pickle.load(f)

print("Total number of filenames:", len(filenames))


model = VGG16(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax")

# Feature extraction function
def feature_extractor_batch(img_paths, model):
    batch_features = []
    for img_path in img_paths:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img)
        result = model.predict(preprocessed_img).flatten()
        batch_features.append(result)
    return batch_features

# Process images in batches
batch_size = 32
features = []
for i in tqdm(range(0, len(filenames), batch_size)):
    batch_filenames = filenames[i:i+batch_size]
    batch_features = feature_extractor_batch(batch_filenames, model)
    features.extend(batch_features)

# Save features to embedding.pkl
pickle.dump(features, open('embedding.pkl', 'wb'))