
# # https://github.com/airctic/mantisshrimp_streamlit -> CHECK THIS READ ME, MODEL AFTER THAT

# https://towardsdatascience.com/demo-your-model-with-streamlit-a76011467dfb
# import streamlit as st
# import pandas as pd
# from io import StringIO
# from PIL import Image, ImageEnhance
# from tensorflow import keras

# st.title('Prostate Cancer Detector')
# st.write('Please upload a whole slide image below')
      
# uploaded_file = st.file_uploader("", type=['tiff', 'jpg','png','jpeg'])
# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
    
#     col1, col2 = st.columns( [0.5, 0.5])
#     with col1:
#         st.markdown('<p style="text-align: center;">Original</p>',unsafe_allow_html=True)
#         st.image(image,width=300)  

#     with col2:
#         st.markdown('<p style="text-align: center;">With Mask</p>',unsafe_allow_html=True)
#         st.image(image,width=300)  

# # model = keras.models.load_model('res50_model_ft.pth')


import io
import os
from PIL import Image
import streamlit as st
import torch
from torchvision import transforms
import wget
import matplotlib.pyplot as plt

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader,Dataset
from torchvision.io import read_image, ImageReadMode

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Image.MAX_IMAGE_PIXELS = 933120000

        
def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data)).convert('RGB')
    else:
        return None
    return uploaded_file

def load_model():
    model = torch.load('res50_model_ft.pth')
    model.eval()
    return model

def load_labels():
    return ['cancer', 'noncancer']

def pre_image(image, model):
   # img = image.convert('RGB')
   # img = image
   # img = Image.open(io.BytesIO(image_data)).convert('RGB')
   # img = read_image(image_path, mode=ImageReadMode.RGB)
   # img = img.convert('RGB')
   # img = Image.open('sample_images/00a26aaa82c959624d90dfb69fcf259c_(512, 512)_noncancer.png').convert('RGB')
   img = Image.open(image_path)
   mean = [0.485, 0.456, 0.406] 
   std = [0.229, 0.224, 0.225]
   transform_norm = transforms.Compose([transforms.ToTensor(), 
   transforms.Resize((224,224)),transforms.Normalize(mean, std)])
   # get normalized image
   img_normalized = transform_norm(img).float()
   img_normalized = img_normalized.unsqueeze_(0)
   # input = Variable(image_tensor)
   img_normalized = img_normalized.to(device)
   # print(img_normalized.shape)
   with torch.no_grad():
      model.eval()  
      output = model(img_normalized)
     # print(output)
      index = output.data.cpu().numpy().argmax()
      # classes = train_ds.classes
      classes = ['cancer', 'noncancer']
      class_name = classes[index]
      return class_name








def main():
    st.title('Pretrained model demo')
    model = load_model()
    categories = load_labels()
    image = load_image()
    result = st.button('Run on image')
    if result:
        st.write('Calculating results...')
        pre_image(model)
        # predict(model, image)
        # torch.argmax(model(image), axis=1)
        # visualize_model_predictions(model, image)



        
if __name__ == '__main__':
    main()

























# uploaded_file = st.file_uploader("Choose a brain MRI ...", type="jpg")
# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded MRI.', use_column_width=True)
#     st.write("")
#     st.write("Classifying...")
#     label = teachable_machine_classification(image, 'brain_tumor_classification.h5')
#     if label == 0:
#         st.write("The MRI scan has a brain tumor")
#     else:
#         st.write("The MRI scan is healthy")
