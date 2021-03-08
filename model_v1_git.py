import streamlit as st
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from skimage import transform
from skimage import exposure
from tensorflow.keras import datasets, layers, models
# import matplotlib as mpl
import requests
from keras.models import model_from_json
from numpy import asarray
from keras.utils.data_utils import get_file


label_df = pd.read_csv("https://raw.githubusercontent.com/lakshmikanth83/Traffic_Sign_Detection/main/label_df_10.csv")  

cwd = r"C:\Users\laksh\Documents\Lakshmikanth\Drexel Classes/Term 6/Capstone Project 2/"

st.title("Traffic Sign Image Classification")


image_path = cwd + "/Test_images/stop.jpg"

# url = st.text_input("label goes here")
# url = "https://images.homedepot-static.com/productImages/a4e315b5-e2b7-4fd9-94a0-c7d51408285d/svn/brady-stock-signs-94143-64_1000.jpg"
# image_path = Image.open(urllib2.urlopen(url))


st.write("## Welcome to Capstone project - 2")
st.write("### Upload images from below labels")
st.write(label_df.labels_cat.value_counts())


url = st.text_input("Input the image URL here")
st.write("[Example link](https://images.homedepot-static.com/productImages/a4e315b5-e2b7-4fd9-94a0-c7d51408285d/svn/brady-stock-signs-94143-64_1000.jpg)")


# json_file = open(cwd + "/model_10.json", 'r')
# loaded_model_json = json_file.read()
# json_file.close()


loaded_model_json = '{"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": [null, 80, 80, 3], "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 47, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "build_input_shape": [null, 80, 80, 3]}, "keras_version": "2.3.0-tf", "backend": "tensorflow"}'

loaded_model = model_from_json(loaded_model_json)


#load weights into new model
loaded_model.load_weights(cwd + "/model_10.h5")

# weights_path = get_file(
#             'model_10_git.h5',
#             'https://github.com/lakshmikanth83/Traffic_Sign_Detection/blob/main/model_10.h5')
# loaded_model.load_weights(weights_path)

st.write("Loaded model from disk")


import requests
def classify_image(url):
    data = Image.open(requests.get(url, stream=True).raw)

    
#     data = Image.open(image_path)
    data = asarray(data)
    image = transform.resize(data, (80, 80))
    image = exposure.equalize_adapthist(image, clip_limit=0.1)

    data = Image.open(requests.get(url, stream=True).raw)

#     data = Image.open(image_path)
    data = asarray(data)
    image1 = transform.resize(data, (80, 80))
    image1 = exposure.equalize_adapthist(image1, clip_limit=0.1)
    
    data = []
    data.append(image)
    data.append(image1)
    data = np.array(data)
    
    predictions = loaded_model.predict(data[1:2])
    score = tf.nn.softmax(predictions[0])
    st.write("Image Given is :")
    st.image(data, width=None)
    
    
#     st.write("Model predicts the image as: ")
    
    md_results = f"## Model predicts the image as: **{label_df['labels_cat'][label_df.labels==np.argmax(score)].head(1).iloc[0]}** ##\n With Confidence of **{100 * np.max(score)}**."

    st.markdown(md_results)
    
#     st.write(label_df['labels_cat'][label_df.labels==np.argmax(score)].head(1).iloc[0])   
#     st.write("With Confidence of")
#     st.write(100 * np.max(score))
    
    

#     plt.imshow(data[1], cmap=mpl.cm.binary)
#     plt.axis("off")
#     plt.show()
             
         
# url = "https://images.homedepot-static.com/productImages/a4e315b5-e2b7-4fd9-94a0-c7d51408285d/svn/brady-stock-signs-94143-64_1000.jpg"
try:    
    classify_image(url)
except ValueError:
    st.error("Please enter a valid input which is in JPG format")    
