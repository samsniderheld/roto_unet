"""
This script leverages the HED (Holistically-Nested Edge Detection) and MiDaS models 
to perform image processing tasks such as generating depth maps and detecting edges 
in images. It contains utility functions and classes to facilitate these tasks, 
including a custom CropLayer class needed for the HED model implementation in OpenCV.

Models:
- MiDaS: used for creating depth maps from input images.
- HED: used for detecting edges in images.

Classes:
- CropLayer: a custom layer to be used in the HED model for cropping images based 
  on certain conditions.

Functions:
- atoi: utility function to convert text to integer if possible.
- natural_keys: function to naturally sort a list of strings containing numbers.
- create_hed: function to process an image using the HED model and return the 
  processed image.
- create_midas: function to process an image using the MiDaS model and return 
  the processed image.
"""

import re

import cv2
import numpy as np
import torch
import tensorflow_addons as tfa
from torchvision import transforms as T

# Load the MiDaS model and utils
MODEL_TYPE = "DPT_Large"  # MiDaS v3 - Large (highest accuracy, slowest inference speed)
midas = torch.hub.load("intel-isl/MiDaS", MODEL_TYPE)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform_m = midas_transforms.dpt_transform

# Load the HED model
prototxt = "hed-edge-detector/deploy.prototxt"
caffemodel = "hed-edge-detector/hed_pretrained_bsds.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)


def atoi(text):
    """Converts text to integer if it represents a digit, otherwise returns the text as is."""
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    Sorts text in a human order.

    Reference:
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]


class CropLayer:
    """A layer for cropping images in the HED model."""
    
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    def getMemoryShapes(self, inputs):
        """Gets the memory shapes for cropping based on the input and target shapes."""
        input_shape, target_shape = inputs[0], inputs[1]
        batch_size, num_channels = input_shape[0], input_shape[1]
        height, width = target_shape[2], target_shape[3]

        self.ystart = (input_shape[2] - target_shape[2]) // 2
        self.xstart = (input_shape[3] - target_shape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batch_size, num_channels, height, width]]

    def forward(self, inputs):
        """Performs forward pass of the crop layer."""
        return [inputs[0][:, :, self.ystart:self.yend, self.xstart:self.xend]]


cv2.dnn_registerLayer('Crop', CropLayer)


def create_hed(img, width, height):
    """Creates an image using the HED model with specified width and height."""
    image = cv2.resize(img, (width, height))

    inp = cv2.dnn.blobFromImage(
        image, scalefactor=1.0, size=(width, height),
        mean=(104.00698793, 116.66876762, 122.67891434),
        swapRB=False, crop=False
    )
    net.setInput(inp)
    out = net.forward()
    out = out[0, 0]
    out = cv2.resize(out, (image.shape[1], image.shape[0]))
    out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    out = (255 * out).astype(np.uint8)
    return out


def create_midas(img):
    """Creates an image using the MiDaS model."""
    input_batch = transform_m(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    output = prediction.cpu().numpy()
    output = (output * 255 / np.max(output)).astype('uint8')
    return output
