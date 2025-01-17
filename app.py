"""
Interactive program for online calculation of the laser beam profile from picture.
"""

import streamlit as st
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from zipfile import ZipFile
import tempfile
import os


# width of displayed picture
width = 100

st.write("# Beam-profiler from picture")


uploaded_files = st.sidebar.file_uploader("Choose grey files with laser beam spot", 
                                          type = ['png', 'jpg', 'bmp', 'tif'], 
                                          accept_multiple_files=True)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()

st.sidebar.write(""" Source code is located [here](https://github.com/Alex-beam/Beam-profiler).   
                 Contact me by [email](mailto:akorom@mail.ru).""")

st.image(uploaded_files, width=width)

option = st.selectbox("How would you like to remove background?",
                        ("None", "Corner", "Histogram"), index=2)


st.write("Processed images:")


imgs = []
imgs_g = []
imgs_df = []

for uploaded_file in uploaded_files:   
    
    img = np.array(Image.open(uploaded_file))
    img = cv2.medianBlur(img,3)

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
    imgs.append(img)

st.image(imgs, width=width)

delta = st.number_input(label = "Select size of final picture in pxl", 
                min_value=10,
                max_value=2000,
                value=400) // 2

st.write("Cropped images with spot in middle:")

for uploaded_file, img in zip(uploaded_files, imgs):   
    # st.image(img, width=width)

    img_g, *_ = cv2.split(img)
    ret, thresh = cv2.threshold(img_g,200,255,cv2.THRESH_BINARY)

    # Calculate the moments of the image
    moments = cv2.moments(thresh)

    # Calculate the center of mass coordinates
    center_x = int(moments['m10'] / moments['m00'])
    center_y = int(moments['m01'] / moments['m00'])
    #print(center_x,center_y)

    # Cropping
    x1 = int(center_x - delta)
    x2 = int(center_x + delta)
    y1 = int(center_y - delta)
    y2 = int(center_y + delta)

    if x1 > 0 and x2 < img_g.shape[1] and y1 > 0 and y2 < img_g.shape[0]:
        img_g = img_g[y1:y2,x1:x2]
        # st.image(img, width=width)
        profile_y = img_g[:, delta]
        profile_x = img_g[delta, :]
    else:
        st.write('Unsuccessful cropping! Image is too small')
        profile_y = img_g[:, center_y]
        profile_x = img_g[center_x, :]
    
    df_x = pd.DataFrame(profile_x, columns=['x'])
    df_y = pd.DataFrame(profile_y, columns=['y'])
    df = pd.concat([df_x,df_y], axis=1)

    imgs_g.append(img_g)
    imgs_df.append(df)

st.image(imgs_g, width=width)


st.write("Colorized images:")
imgs_cmap = []

for img_g in imgs_g:
    
    img_cmap = cv2.cvtColor(cv2.applyColorMap(img_g, 2), cv2.COLOR_BGR2RGB)
    imgs_cmap.append(img_cmap)

st.image(imgs_cmap, width=width)

imgs_cmap = []

for img_g in imgs_g:
    
    img_cmap = cv2.applyColorMap(img_g, 2)
    imgs_cmap.append(img_cmap)


# Creating new images and profiles and and to zip file 
extension = st.selectbox("Choose the extension of new image files", ['png', 'jpg', 'tif'])

with tempfile.TemporaryDirectory() as temp_dir_name:
    for uploaded_file, img_g, img_cmap, df in zip(uploaded_files, imgs_g, imgs_cmap, imgs_df):  

        cv2.imwrite(temp_dir_name + r"/" + '.'.join(uploaded_file.name.split('.')[:-1]) + "_grey" + "." + extension, img_g)
        cv2.imwrite(temp_dir_name + r"/" + '.'.join(uploaded_file.name.split('.')[:-1]) + "_color" + "." + extension, img_cmap)
        df.to_csv(temp_dir_name + r"/" + '.'.join(uploaded_file.name.split('.')[:-1]) + "_profiles" + ".csv", encoding='utf-8')

    path = os.getcwd()
    os.chdir(temp_dir_name)

    file_paths = []

    for f in os.scandir():
        if f.is_file():
            file_paths.append(f.path)

    # writing files to a zipfile 
    with ZipFile(temp_dir_name + r"/profiles.zip", "w") as zip: 
        # writing each file one by one 
        for file in file_paths: 
            zip.write(file)


    with open(temp_dir_name + r"/profiles.zip", "rb") as file:
        btn = st.download_button(
                label="Download new pictures and profiles in zip",
                data=file,
                file_name="profiles.zip",
                )
    
    os.chdir(path)
        
     