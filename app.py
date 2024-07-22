"""
Interactive program for online calculation of the beam profile from picture.
"""

import streamlit as st
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from zipfile import ZipFile


def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")


print('Go!')
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
                    value=400) // 2


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
        img = img[y1:y2,x1:x2]
        st.image(img, width=width)
    else:
        st.write('Image is too small')
    
    # Display profile
    profile_y = img[:, delta, 1]
    profile_x = img[delta, :, 1]
    profile_y = profile_y[..., None]
    profile_x = profile_x[..., None]
    data = np.hstack((profile_x, profile_y))

    df = pd.DataFrame(data, columns=['x','y'])
    st.line_chart(df)

    cv2.imwrite("image\\Beam_profile.png", img)


    st.download_button(label='Download data frame', data=convert_df(df), file_name='df.csv',  mime='text/csv')
    
    with open("image\\Beam_profile.png", "rb") as file:
        btn = st.download_button(
                label="Download image",
                data=file,
                file_name="Beam_profile.png",
                mime="image/png"
            )
        
    # path to folder which needs to be zipped 
    # directory = './image'
  
    # calling function to get all file paths in the directory 
    file_paths = ["image\\Beam_profile.png"]
  
    # printing the list of all files to be zipped 
    # print('Following files will be zipped:') 
    # for file_name in file_paths: 
    #     print(file_name) 
  
    # writing files to a zipfile 
    with ZipFile("temp\\my_python_files.zip","w") as zip: 
        # writing each file one by one 
        for file in file_paths: 
            zip.write(file) 
  
    print('All files zipped successfully!')

    
    with open("temp\\my_python_files.zip", "rb") as file:
        btn = st.download_button(
                label="Download zip",
                data=file,
                file_name="my_python_files.zip",
                )         