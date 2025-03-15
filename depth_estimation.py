import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from midas.model_loader import load_model

def preprocess_image(image):
    # Convert NumPy array (BGR) to PIL Image (RGB)
    image = Image.fromarray(image)

    transform = Compose([
        Resize((384, 384)),
        ToTensor(),
        Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(image).unsqueeze(0)


def estimate_depth(model, frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = preprocess_image(frame_rgb)

    with torch.no_grad():
        depth_map = model(frame_tensor)

    depth_map = depth_map.squeeze().cpu().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())  # Normalize
    return depth_map
