"""
Interactive program for online calculation of the beam profile from picture.
"""

import streamlit as st
import cv2
from PIL import Image
import numpy as np


st.write("# Beam-profiler from picture")

uploaded_files = st.sidebar.file_uploader("Choose files (png, jpg, tiff)", 
                                          type = ['png', 'jpg', 'bmp', 'tif'], 
                                          accept_multiple_files=True)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    # st.write("filename:", uploaded_file.name)
    st.image(uploaded_file, width= 100)

width = 300
if uploaded_files:
    st.image(uploaded_files, width=width)
    img = np.array(Image.open(uploaded_files[0]))
    img = cv2.medianBlur(img,3)

    option = st.selectbox("How would you like to remove background?",
                          ("None", "Corner", "Histogram"), index=2)

    match option:
        case "None":
            pass
        case "Corner":
            # Mean in corner
            background = (img[:10,:10].mean(dtype = "uint64")  + 1).astype("uint8")
            img[img<background] = background
        case "Histogram":
            # First local minimum
            hist = cv2.calcHist([img],[0],None,[256],[0,256])
            minima = np.where((hist[1:-1] < hist[0:-2]) * (hist[1:-1] < hist[2:]))[0] + 1
            if any(minima):
                background = [minima[0]]
                img[img<background] = background    


    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    st.image(img, width=width)

    delta = st.number_input(label = "Select size of final picture in pxl", 
                    min_value=10,
                    max_value=2000,
                    value=400) / 2


    img_g, *_ = cv2.split(img)
    ret, thresh = cv2.threshold(img_g,200,255,cv2.THRESH_BINARY)

    # Calculate the moments of the image
    moments = cv2.moments(thresh)

    # Calculate the center of mass coordinates
    center_x = int(moments['m10'] / moments['m00'])
    center_y = int(moments['m01'] / moments['m00'])
    #print(center_x,center_y)

    # Cropping
    #print(img_g.shape)
    x1 = int(center_x - delta)
    x2 = int(center_x + delta)
    y1 = int(center_y - delta)
    y2 = int(center_y + delta)

    if x1 > 0 and x2 < img_g.shape[1] and y1 > 0 and y2 < img_g.shape[0]:
        img = img[y1:y2,x1:x2]
        st.image(img, width=width)
    else:
        st.write("Image is too small")
    
    # Display profile
    profile_y = img_g[center_y]
    profile_x = img_g[:,center_x]

    st.line_chart(profile_x)
    st.line_chart(profile_y)