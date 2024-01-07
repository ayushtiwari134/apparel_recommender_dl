import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from numpy.linalg import norm
import pickle
from sklearn.neighbors import NearestNeighbors



# resnet model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = Sequential(
    [
        model,  # uptil here the model is the same as that which we imported
        GlobalMaxPooling2D(),  # the new GlobalMaxPooling2D layer
    ]
)

features_list = pickle.load(open("features.pkl", "rb"))


def predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    features = features / norm(features)
    features = features.flatten()
    return features

path = 'images\1164.jpg'

neighbours = NearestNeighbors(n_neighbors=5, algorithm="brute", metric="euclidean")
neighbours.fit(features_list)

dist, indices = neighbours.kneighbors([predict(r'images\1164.jpg')])

print(indices)
