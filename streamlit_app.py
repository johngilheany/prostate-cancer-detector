import streamlit as st
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import io
from PIL import Image

model = keras.models.load_model('models/model1_checkpoint.keras')

if 'image' not in st.session_state:
    st.session_state['image'] = '00a7fb880dc12c5de82df39b30533da9.tiff'

title_html = '''
    <h1 style="color: white">Prostate Cancer Detector.</h1>
    '''
# st.markdown(':xx:'*96 + title_html, unsafe_allow_html=True)
# st.subheader("Please Upload a Whole Slide Image")

placeholder = st.empty()

def show_image():   
    placeholder.image(st.session_state['image'])

show_image()

hook_html = '''
    <h3 style="color: white">Please Upload a Whole Slide Image</h3>
    '''
st.markdown(hook_html, unsafe_allow_html=True)

uploaded_file = st.file_uploader('', on_change=show_image())

if uploaded_file:
    bytes_data = uploaded_file.getvalue()
    nparray = np.asarray(Image.open(io.BytesIO(bytes_data)).resize((256,256), Image.ANTIALIAS) )
    np_reshaped = nparray.reshape(1, 256, 256, 3)
    pred = np.argmax(model.predict(np_reshaped), axis = 1)
    placeholder.image(bytes_data)
    if pred[0] == 0:
        result = '''
        <h3 style="color: green; text-align: center">This is cancerous</h3>
        '''
        st.markdown(result, unsafe_allow_html=True)
    else:
        result = '''
        <h3 style="color: red; text-align: center">This is not cancerous</h3>
        '''
        st.markdown(result, unsafe_allow_html=True)
