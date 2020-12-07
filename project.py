import tensorflow as tf
import streamlit as st
import numpy as np
import pandas as pd 
from PIL import ImageOps, Image


def import_and_predict(image_data, model):
    size = (128,128)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    
    return prediction

def load_model():
    path = 'model_final.h5'
    model = tf.keras.models.load_model(path)

    return model  

st.markdown("""<style>body{background-color: #262626;color:#3399ff} .sidebar-content{color: black}</style>""",unsafe_allow_html=True)

st.sidebar.title("Classification Image")
st.sidebar.title("Choose a File")
file = st.sidebar.file_uploader("FILE", type=['jpg','png'])
button = st.sidebar.button("Predict")
st.title("Malaria Cell Image")

model = load_model()

if file is None:
    st.title('Please upload an image file')
else:
    image = Image.open(file)
    st.image(image, width=400, use_column_width=False)
    if button:
        predictions = import_and_predict(image, model)
        class_name = ['Parasitized', 'Uninfected']
        st.text('This image most likely is: ')
        st.success(class_name[np.argmax(predictions)])