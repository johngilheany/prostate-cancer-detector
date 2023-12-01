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
    st.session_state['image'] = 'hot_dogs_in_a_row.jpg'


title_html = '''
    <h1 style="color: white">The Hot Dog showdown. Wiener takes all.</h1>
    '''
#title = st.title('The Hot Dog showdown. Wiener takes all.')
st.markdown(':hotdog:'*96 + title_html, unsafe_allow_html=True)
st.subheader("Go ahead and try. We'll find the frankfurter.")

placeholder = st.empty()

def show_image():   
    placeholder.image(st.session_state['image'])


#image = st.image(starter_image)
# starter_image = 'hot_dogs_in_a_row.jpg'
show_image()

#if model.predict
hook_html = '''
    <h3 style="color: white">Upload your hot dog (or not) here, weenie</h3>
    '''
st.markdown(hook_html, unsafe_allow_html=True)

uploaded_file = st.file_uploader('', on_change=show_image())

if uploaded_file:
    # bytes_data = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    bytes_data = uploaded_file.getvalue()
    



    nparray = np.asarray(Image.open(io.BytesIO(bytes_data)).resize((256,256), Image.ANTIALIAS) )


    # np_reshaped = tf.keras.preprocessing.image.smart_resize(
    # nparray, (256,256), interpolation='bilinear'
    # )

    np_reshaped = nparray.reshape(1, 256, 256, 3)
    
    # import pdb
    # pdb.set_trace()
    pred = np.argmax(model.predict(np_reshaped), axis = 1)
    # probs = list(model.predict_proba(np_reshaped))
    # st.text(pred)
    #st.text(probs)
    #TODO: pickled model.predict(np_reshaped)
    #if probs >= 0.5:
       #st.text()
    placeholder.image(bytes_data)
    if pred[0] == 0:
        result = '''
        <h3 style="color: green; text-align: center">This is definitely a DOG</h3>
        '''
        st.markdown(result, unsafe_allow_html=True)
    else:
        result = '''
        <h3 style="color: red; text-align: center">This ain't no dog</h3>
        '''
        st.markdown(result, unsafe_allow_html=True)


# st.image(main_image)
# if image_file:
#     image = st.image(image_file)
# TODO

