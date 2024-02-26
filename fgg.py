import pickle
import numpy as np

# Load features from embedding.pkl
features = pickle.load(open('embedding.pkl', 'rb'))
num_feature_vectors = len(features)
print("Number of feature vectors:", num_feature_vectors)

# Check format
if isinstance(features, list):
    print("Features are stored as a list.")
elif isinstance(features, np.ndarray):
    print("Features are stored as a numpy array.")
else:
    print("Unexpected format for features.")

# Check length
num_features = len(features)
expected_num_features = 266  # Set the expected number of feature vectors based on the number of images processed
if num_features == expected_num_features:
    print("Number of feature vectors matches the expected number.")
else:
    print("Number of feature vectors does not match the expected number.")

# Inspect a few feature vectors (optional)
if num_features > 0:
    print("Example feature vector shape:", features[0].shape)
    print("Example feature vector:", features[0])
else:
    print("No feature vectors extracted.")
