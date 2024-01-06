import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras import Sequential
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn import set_config
from numpy.linalg import norm
import pickle
#resnet model
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input

#load model
model = pickle.load(open('model.pkl', 'rb'))

# def reshape_image_to_input(path):
#     img = image.load_img(path,target_size=(224,224))
#     img_array = image.img_to_array(img)
#     reshaped_img = np.expand_dims(img_array,axis=0)
#     preprocessed_img_arr = preprocess_input(reshaped_img)
#     return preprocessed_img_arr

img = 'images/1563.jpg'

arr = model.fit_transform(img)

print(arr)
