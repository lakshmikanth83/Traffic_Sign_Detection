import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from skimage import transform
from skimage import exposure
from tensorflow.keras import datasets, layers, models
import matplotlib as mpl
import requests
from keras.models import model_from_json
from numpy import asarray


label_df = pd.read_csv(r"C:\Users\laksh\Documents/Lakshmikanth/Drexel Classes/Term 6/Capstone Project 2/label_df_10.csv")  
  

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


json_file = open(cwd + "/model_10.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
st.write("Loading model completed")
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(cwd + "/model_10.h5")
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