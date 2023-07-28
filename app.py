# Native imports
import os
import argparse
import math
import numbers
from collections import namedtuple, defaultdict
import enum

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms

# Visualize imports
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# App related
import streamlit as st
from predict import *

# Parameters for the model
config = {
    "model_name": 'RESNET50',
    "pretrained_weights": 'IMAGENET',
    "pyramid_size": 4,
    "pyramid_ratio": 2.1,
    "num_gradient_ascent_iterations": 15,
    "lr": 0.05,
    "img_width" : 600,
    "input" : "data/input/test.jpg",
    "layers_to_use" : ["layer3"],
    "should_display" : False,
    "spatial_shift_size" : 40,
    "smoothing_coefficient" : 0.5,
    "use_noise " : False}

config['dump_dir'] = os.path.join(OUT_IMAGES_PATH, f'{config["model_name"]}_{config["pretrained_weights"]}')
config['input'] = os.path.basename(config['input'])  # handle absolute and relative paths

# Markdown
st.title("My Deep Dream Project")
st.subheader("Upload your Image below")

st.write("#")
st.write("#")

uploaded_file = st.file_uploader("Only JPG, PNG, and JPEG formats supported as of today...", type=['jpg', 'png'])


if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    input_image = cv.imdecode(file_bytes, 1)[ : , : , :  : -1]
    st.subheader("Uploaded Image looks like this")
    st.image(input_image, channels = "RGB")
    

    st.write("Received Image succesfully")
    st.write("Dimensions are", input_image.shape)
    #st.write("Type of Input Image", type(input_image))

    if st.button("Click here to generate!"):
        st.write("Generating")
        output_image = deep_dream_static(config, input_image)


        st.write("# Generated Output")
        st.image(output_image, channels = "RGB")