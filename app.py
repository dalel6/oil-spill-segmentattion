import streamlit as st
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss

# Define your SegmentationModel class
class SegmentationModel(nn.Module):
    def __init__(self, encoder='timm-efficientnet-b0', weights='imagenet'):
        super(SegmentationModel, self).__init__()
        self.arc = smp.Unet(
            encoder_name=encoder, 
            encoder_weights=weights, 
            in_channels=3, 
            classes=1, 
            activation=None
        )

    def forward(self, images, masks=None):
        logits = self.arc(images)
        if masks is not None:
            loss1 = DiceLoss(mode='binary')(logits, masks)
            loss2 = nn.BCEWithLogitsLoss()(logits, masks)
            return logits, loss1 + loss2
        return logits

# Global transform
preprocess_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load model function
def load_model(model_path, encoder, weights):
    model = SegmentationModel(encoder, weights)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocess image function
def preprocess_image(image):
    image = np.array(image)
    
    # Calculate padding
    height, width, _ = image.shape
    pad_height = (32 - height % 32) % 32
    pad_width = (32 - width % 32) % 32

    # Pad the image
    padded_image = cv2.copyMakeBorder(image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Apply transforms
    padded_image = preprocess_transform(padded_image).unsqueeze(0)
    return padded_image


# Segment image function
def segment_image(image, model):
    with torch.no_grad():
        logits = model(image)
        pred_mask = torch.sigmoid(logits)
        output = (pred_mask > 0.5).float()
        return output

# Load your trained model
model = load_model('./bestmodel.pt', 'timm-efficientnet-b0', 'imagenet')

# Streamlit UI
st.title('Oil Spill Segmentation App')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Segmenting...")

    processed_image = preprocess_image(image)
    segmented_image = segment_image(processed_image, model)

    # Convert the segmented image to a displayable format
    segmented_image_np = segmented_image.squeeze(0).cpu().numpy()[0]
    st.image(segmented_image_np, use_column_width=True)
    st.markdown("<h2 style='color: red;'>There is a lot of oil spill, so you can't swim even fishing</h2>", unsafe_allow_html=True)



