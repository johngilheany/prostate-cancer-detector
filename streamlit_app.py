# import libraries
from PIL import Image
from torchvision import models, transforms
import torch
import streamlit as st

Image.MAX_IMAGE_PIXELS = 933120000

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
        # st.write('<p style="font-size:32px; color:black;">This sample is benign</p>', unsafe_allow_html=True)
        st.write('This sample is benign')
    if label == 'cancer':
        # st.write('<p style="font-size:32px; color:black;">This sample is cancerous</p>', unsafe_allow_html=True)
        st.write('This sample is cancerous')

# def make_tiles_from_image(tile_size, level, output_folder=None):
#     image_id = sample.loc['image_id']
#     data_provider = sample.loc['data_provider']
#     slide = openslide.OpenSlide(os.path.join(data_dir, image_id+'.tiff'))
#     mask = openslide.OpenSlide(os.path.join(mask_dir, image_id+'_mask.tiff'))
#     tiles = get_tile_locations_from_slide(slide, tile_size=256, N=36, level=1)
    
#     if output_folder == None:
#         output_folder = 'data' + image_id
#         if not os.path.exists(output_folder):
#             os.makedirs(output_folder)
        
#     for tile in tiles:
#         tile_coordinate = f'''({str(tile['xloc'])}, {str(tile['yloc'])})'''
#         image_tile_array = tile_array(slide, tile, tile_size, level)
#         mask_tile_array = tile_array(mask, tile, tile_size, level)
#         label = classify_as_cancerous(mask_tile_array, data_provider)
#         filename = f'{image_id}_{tile_coordinate}_{label}'
#         save_img(image_tile_array, filename, output_folder)
    
#     return os.path.join(data_dir, image_id+'.tiff'), output_folder

# tile_size = 256 # tile size
# level = 1 # level
# pxsize = 227
# img_path, tiles_path = make_tiles_from_image(tile_size, level, output_folder=None) # this will create a subfolder inside the specified output_folder containing the tiles 
# tiles = glob.glob(os.path.join(tiles_path, '*cancer.png')) 
# labels = [int('noncancer' not in tile) for tile in tiles]

# X = preprocess(Image.open(tiles[0]).convert('RGB')).reshape(1,3,pxsize,pxsize) # convert to RGB to drop alpha channel of png 
# for tile in tiles[1:]:
#     X = torch.cat((preprocess(Image.open(tile).convert('RGB')).reshape(1,3,pxsize,pxsize), X), axis=0)

# Y = torch.Tensor(labels).type(torch.LongTensor)
# probs = nn.Softmax()(res50(X))[:,1].detach().numpy()
# slide = openslide.OpenSlide(img_path)

# # Overlap red mask
# content = np.asarray(slide.read_region((0,0), 1, slide.level_dimensions[1])).copy()

# for i, tile in enumerate(tiles):
#     x, y = [int(w) for w in re.findall(r'\d+', tile.split("_")[2])]
#     detect_white = np.sum(content[y:(y+tile_size),x:(x+tile_size),:], axis=2) < 900
#     if probs[i] > 0.4:
#         content[y:(y+tile_size),x:(x+tile_size),1] = (1-detect_white)*content[y:(y+tile_size),x:(x+tile_size),1] + detect_white*(1-probs[i])*255
#         content[y:(y+tile_size),x:(x+tile_size),2] = (1-detect_white)*content[y:(y+tile_size),x:(x+tile_size),2] + detect_white*(1-probs[i])*255
#         # content[y:(y+tile_size),x:(x+tile_size),1] = (1-detect_white)*content[y:(y+tile_size),x:(x+tile_size),2] + detect_white*255
#         # content[y:(y+tile_size),x:(x+tile_size),3] = (1-detect_white)*content[y:(y+tile_size),x:(x+tile_size),3] + detect_white*(0.99*255 + 0.0*probs[i]*255)

# content_green = content.copy()
# Image.fromarray(content)



















