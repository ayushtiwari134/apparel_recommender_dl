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
import streamlit as st
import os
from PIL import Image


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
filenames = pickle.load(open("filenames.pkl", "rb"))

def predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    features = features / norm(features)
    features = features.flatten()
    return features


def recommend(features_list, feat):
    neighbours = NearestNeighbors(n_neighbors=5, algorithm="brute", metric="euclidean")
    neighbours.fit(features_list)
    distances, indices = neighbours.kneighbors([feat])
    return indices


# streamlit code
st.title("Apparel Recommendation System")

uploaded = st.file_uploader("Choose an image...")


def save_uploaded_file(uploadedfile):
    try:
        with open(os.path.join("uploads", uploadedfile.name), "wb") as f:
            f.write(uploadedfile.getbuffer())
    except:
        print("0")


if uploaded is not None:
    save_uploaded_file(uploaded)
    display_image = Image.open(uploaded)
    st.image(display_image)
    feat = predict(os.path.join("uploads", uploaded.name))
    indices = recommend(features_list, feat)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if filenames[indices[0][0]]:
            st.image(filenames[indices[0][0]])
        else:
            st.write("Image was not uploaded")
    with col2:
        if filenames[indices[0][1]]:
            st.image(filenames[indices[0][1]])
        else:
            st.write("Image was not uploaded")
    with col3:
        if filenames[indices[0][2]]:
            st.image(filenames[indices[0][2]])
        else:
            st.write("Image was not uploaded")
    with col4:
        if filenames[indices[0][3]]:
            st.image(filenames[indices[0][3]])
        else:
            st.write("Image was not uploaded")
    with col5:
        if filenames[indices[0][4]]:
            st.image(filenames[indices[0][4]])
        else:
            st.write("Image was not uploaded")
    

else:
    st.error("Please upload an image file")
