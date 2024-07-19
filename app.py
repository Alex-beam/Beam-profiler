"""
Interactive program for online calculation of the beam profile from picture.
"""

import streamlit as st
import cv2
from PIL import Image
import numpy as np

st.write("# Beam-profiler from picture")

uploaded_files = st.sidebar.file_uploader("Choose files (png, jpg, tiff)", 
                                          type = ['png', 'jpg'], 
                                          accept_multiple_files=True)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    # st.write("filename:", uploaded_file.name)
    st.image(uploaded_file, width= 100)
if uploaded_files:
    st.image(uploaded_files)
    st.write(uploaded_files[0])
    img = np.array(Image.open(uploaded_files[0]))
    print(img.shape)
    img = cv2.medianBlur(img,3)
    st.image(img)
