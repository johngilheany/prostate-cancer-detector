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


# def load_image():
#     uploaded_file = st.file_uploader(label='Pick an image to test')
#     if uploaded_file is not None:
#         image_data = uploaded_file.getvalue()
#         st.image(image_data)
#         return Image.open(io.BytesIO(image_data))
#     else:
#         return None


# def load_model():
#     model = torch.load('res50_model_ft.pth')
#     model.eval()
#     return model


# # def load_labels():
# #     labels_path = ['cancer', 'noncancer']
# #     labels_file = os.path.basename(labels_path)
# #     if not os.path.exists(labels_file):
# #         wget.download(labels_path)
# #     with open(labels_file, "r") as f:
# #         categories = [s.strip() for s in f.readlines()]
# #         return categories

# def load_labels():
#     return ['cancer', 'noncancer']

# # def predict(model, categories, image):
# #     preprocess = transforms.Compose([
# #         transforms.Resize(256),
# #         transforms.CenterCrop(224),
# #         transforms.ToTensor(),
# #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# #     ])
# #     input_tensor = preprocess(image)
# #     input_batch = input_tensor.unsqueeze(0)

# #     with torch.no_grad():
# #         output = model(input_batch)

# #     probabilities = torch.nn.functional.softmax(output[0], dim=0)

# #     top5_prob, top5_catid = torch.topk(probabilities, 5)
# #     for i in range(top5_prob.size(0)):
# #         st.write(categories[top5_catid[i]], top5_prob[i].item())

# def predict(model, image):
#     # pred = torch.max(output.data, 1)
#     pred_labels = torch.argmax(model(image), axis=1)
#     return pred_labels


# def main():
#     st.title('Pretrained model demo')
#     model = load_model()
#     categories = load_labels()
#     image = load_image()
#     result = st.button('Run on image')
#     if result:
#         st.write('Calculating results...')
#         # predict(model, image)
#         torch.argmax(model(image), axis=1)

# if __name__ == '__main__':
#     main()




"""Create an Image Classification Web App using PyTorch and Streamlit."""
# import libraries
from PIL import Image
from torchvision import models, transforms
import torch
import streamlit as st

# set title of app
st.title("Simple Image Classification Application")
st.write("")

# enable users to upload images for the model to make predictions
file_up = st.file_uploader("Upload an image", type = "jpg")


def predict(image):
    """Return top 5 predictions ranked by highest probability.

    Parameters
    ----------
    :param image: uploaded image
    :type image: jpg
    :rtype: list
    :return: top 5 predictions ranked by highest probability
    """
    # create a ResNet model
    resnet = models.resnet101(pretrained = True)

    # transform the input image through resizing, normalization
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
            )])

    # load the image, pre-process it, and make predictions
    img = Image.open(image)
    batch_t = torch.unsqueeze(transform(img), 0)
    resnet.eval()
    out = resnet(batch_t)

    with open('imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    # return the top 5 predictions ranked by highest probabilities
    prob = torch.nn.functional.softmax(out, dim = 1)[0] * 100
    _, indices = torch.sort(out, descending = True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]


if file_up is not None:
    # display image that user uploaded
    image = Image.open(file_up)
    st.image(image, caption = 'Uploaded Image.', use_column_width = True)
    st.write("")
    st.write("Just a second ...")
    labels = predict(file_up)

    # print out the top 5 prediction labels with scores
    for i in labels:
        st.write("Prediction (index, name)", i[0], ",   Score: ", i[1])














