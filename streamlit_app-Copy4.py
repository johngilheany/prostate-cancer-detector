
# # # https://github.com/airctic/mantisshrimp_streamlit -> CHECK THIS READ ME, MODEL AFTER THAT

# # https://towardsdatascience.com/demo-your-model-with-streamlit-a76011467dfb
# # import streamlit as st
# # import pandas as pd
# # from io import StringIO
# # from PIL import Image, ImageEnhance
# # from tensorflow import keras

# # st.title('Prostate Cancer Detector')
# # st.write('Please upload a whole slide image below')
      
# # uploaded_file = st.file_uploader("", type=['tiff', 'jpg','png','jpeg'])
# # if uploaded_file is not None:
# #     image = Image.open(uploaded_file)
    
# #     col1, col2 = st.columns( [0.5, 0.5])
# #     with col1:
# #         st.markdown('<p style="text-align: center;">Original</p>',unsafe_allow_html=True)
# #         st.image(image,width=300)  

# #     with col2:
# #         st.markdown('<p style="text-align: center;">With Mask</p>',unsafe_allow_html=True)
# #         st.image(image,width=300)  

# # # model = keras.models.load_model('res50_model_ft.pth')


# import io
# import os
# from PIL import Image
# import streamlit as st
# import torch
# from torchvision import transforms
# import wget
# import matplotlib.pyplot as plt

# import torch
# import torchvision.models as models
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# from torch.utils.data import DataLoader,Dataset
# from torchvision.io import read_image, ImageReadMode

# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Image.MAX_IMAGE_PIXELS = 933120000

        
# def load_image():
#     uploaded_file = st.file_uploader(label='Pick an image to test')
#     if uploaded_file is not None:
#         image_data = uploaded_file.getvalue()
#         st.image(image_data)
#         return Image.open(io.BytesIO(image_data)).convert('RGB')
#     else:
#         return None
#     return uploaded_file

# def load_model():
#     model = torch.load('res50_model_ft.pth')
#     model.eval()
#     return model

# def load_labels():
#     return ['cancer', 'noncancer']

# def pre_image(image,model):
#    img = Image.open(image)
#    # img = read_image(image_path, mode=ImageReadMode.RGB)
#    # img = img.convert('RGB')
#    mean = [0.485, 0.456, 0.406] 
#    std = [0.229, 0.224, 0.225]
#    transform_norm = transforms.Compose([transforms.ToTensor(), 
#    transforms.Resize((224,224)),transforms.Normalize(mean, std)])
#    # get normalized image
#    img_normalized = transform_norm(img).float()
#    img_normalized = img_normalized.unsqueeze_(0)
#    # input = Variable(image_tensor)
#    img_normalized = img_normalized.to(device)
#    # print(img_normalized.shape)
#    with torch.no_grad():
#       model.eval()  
#       output = model(img_normalized)
#      # print(output)
#       index = output.data.cpu().numpy().argmax()
#       # classes = train_ds.classes
#       classes = ['cancer', 'noncancer']
#       class_name = classes[index]
#       return class_name






# def main():
#     st.title('Pretrained model demo')
#     model = load_model()
#     categories = load_labels()
#     image = load_image()
#     result = st.button('Run on image')
#     if result:
#         st.write('Calculating results...')
#         pre_image(image,model)
#         # predict(model, image)
#         # torch.argmax(model(image), axis=1)
#         # visualize_model_predictions(model, image)



        
# if __name__ == '__main__':
#     main()

"""Create an Image Classification Web App using PyTorch and Streamlit."""
# import libraries
from PIL import Image
from torchvision import models, transforms
import torch
import streamlit as st

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set title of app
st.title("Simple Image Classification Application")
st.write("")

# enable users to upload images for the model to make predictions
file_up = st.file_uploader("Upload an image", type = ["png", 'jpg', 'tiff'])


def predict(image):
    # create a ResNet model
    # resnet = models.resnet101(pretrained = True)
    model = torch.load('res50_model_ft2.pth')
    model.eval()

    # # transform the input image through resizing, normalization
    # transform = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         mean = [0.485, 0.456, 0.406],
    #         std = [0.229, 0.224, 0.225]
    #         )])

    # load the image, pre-process it, and make predictions
    img = Image.open(image).convert('RGB')
       # img = Image.open(image_path).convert('RGB')
    # img = read_image(image_path, mode=ImageReadMode.RGB)
    # img = img.convert('RGB')
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
    # batch_t = torch.unsqueeze(transform(img), 0)
    # resnet.eval()
    # out = resnet(batch_t)

#     with open('imagenet_classes.txt') as f:
#         classes = [line.strip() for line in f.readlines()]

#     # return the top 5 predictions ranked by highest probabilities
#     prob = torch.nn.functional.softmax(out, dim = 1)[0] * 100
#     _, indices = torch.sort(out, descending = True)
#     return [(classes[idx], prob[idx].item()) for idx in indices[0][:1]]




if file_up is not None:
    # display image that user uploaded
    image = Image.open(file_up)
    st.image(image, caption = 'Uploaded Image.', use_column_width = True)
    st.write("")
    st.write("Just a second ...")
    labels = predict(file_up)
    st.write(labels)

    # # print out the top 5 prediction labels with scores
    # for i in labels:
    #     st.write("Prediction (index, name)", i[0], ",   Score: ", i[1])























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
