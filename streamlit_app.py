# import libraries
from PIL import Image
from torchvision import models, transforms
import torch
import streamlit as st

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set title of app
st.title('Prostate Cancer Detector')
st.write("")
st.write('Please upload a whole slide image below')

# enable users to upload images for the model to make predictions
file_up = st.file_uploader("Upload an image", type = ["png", 'jpg', 'tiff'])


def predict(image):
    # create a ResNet model
    # resnet = models.resnet101(pretrained = True)
    model = torch.load('res50_model_ft2.pth')
    model.eval()

    # load the image, pre-process it, and make predictions
    img = Image.open(image).convert('RGB')
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
        index = output.data.cpu().numpy().argmax()
        classes = ['cancer', 'noncancer']
        class_name = classes[index]
        return class_name

if file_up is not None:
    # display image that user uploaded
    image = Image.open(file_up)
    st.image(image, caption = 'Uploaded Image.', use_column_width = True)
    st.write("")
    label = predict(file_up)
    if label == 'noncancer':
        st.write('This sample is benign')
    if label == 'cancer':
        st.write('This sample is cancerous')























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
