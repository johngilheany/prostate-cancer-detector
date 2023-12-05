# import libraries
from PIL import Image
from torchvision import models, transforms
import torch
import streamlit as st

Image.MAX_IMAGE_PIXELS = 933120000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

# set title of app
st.title('Prostate Cancer Detector')
st.write("")
st.write('Please upload a tissue sample below')

# enable users to upload images for the model to make predictions
file_up = st.file_uploader("Upload an image", type = ["png", 'jpg', 'tiff'])

def predict(image):
    # Load ResNet model
    model = torch.load('model.pth', map_location = torch.device('cpu'))
    model.eval()

    # load the image, pre-process it, and make predictions
    img = Image.open(image).convert('RGB')
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]
    transform_norm = transforms.Compose([transforms.ToTensor(), 
    transforms.Resize((224,224)),transforms.Normalize(mean, std)])
    img_normalized = transform_norm(img).float()
    img_normalized = img_normalized.unsqueeze_(0)
    img_normalized = img_normalized.to(device)
    with torch.no_grad():
        model.eval()  
        output = model(img_normalized)
        index = output.data.cpu().numpy().argmax()
        classes = ['cancer', 'noncancer']
        class_name = classes[index]
        return class_name

if file_up is not None:
    # display image that user uploaded
    image = Image.open(file_up)
    # st.image(image,width=300) use_column_width = True, 
    st.image(image, caption = 'Uploaded Image.', width=300)
    st.write("")
    label = predict(file_up)
    if label == 'noncancer':
        st.markdown('<p style="text-align: left;">This sample is <strong>benign</strong</p>',unsafe_allow_html=True)
    if label == 'cancer':
        st.markdown('<p style="text-align: left;">This sample is <strong>cancerous</strong</p>',unsafe_allow_html=True)
