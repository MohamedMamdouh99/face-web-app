from settings import *
import os
import shutil
import pickle
import datetime
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import uuid

expected_password = '111'
IMAGE_SIZE = 100  # Define the desired image size
COLS_ENCODE = ['Encode1', 'Encode2', 'Encode3']  # Define the column names for face encodings
COLS_INFO = ['Name']  # Define the column names for face information

user_color = '#000000'
title_webapp = "Face Attendance Webapp"

html_temp = f"""
            <div style="background-color:{user_color};padding:12px">
            <h1 style="color:white;text-align:center;">{title_webapp}
            </h1>
            </div>
            """
st.markdown(html_temp, unsafe_allow_html=True)

###################### Defining Static Paths ###################
if st.sidebar.button('Click to Clear out all the data'):
    ## Clearing Visitor Database
    shutil.rmtree(VISITOR_DB, ignore_errors=True)
    os.mkdir(VISITOR_DB)
    ## Clearing Visitor History
    shutil.rmtree(VISITOR_HISTORY, ignore_errors=True)
    os.mkdir(VISITOR_HISTORY)

if not os.path.exists(VISITOR_DB):
    os.mkdir(VISITOR_DB)

if not os.path.exists(VISITOR_HISTORY):
    os.mkdir(VISITOR_HISTORY)

########################################################################################################################
def main():
    selected_menu = option_menu(None,
        ['Visitor Validation', 'History', 'Add to Database'],
        icons=['camera', "clock-history", 'person-plus'],
        ## icons from website: https://icons.getbootstrap.com/
        menu_icon="cast", default_index=0, orientation="horizontal")

    if selected_menu == 'Visitor Validation':
        ## Generates a Random ID for image storage
        visitor_id = uuid.uuid1()

        ## Reading Camera Image
        img_file_buffer = st.camera_input("Take a picture")

        if img_file_buffer is not None:
            bytes_data = img_file_buffer.getvalue()

            # convert image from opened file to np.array
            image_array = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            ## Validating Image
            # Convert the image to grayscale
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

            # Load the Haar cascade files for face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            # Detect faces in the grayscale image
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            ## Number of Faces identified
            max_faces = len(faces)

            if max_faces > 0:
                dataframe_new = pd.DataFrame()

                ## Iterating faces one by one
                for face_idx, (x, y, w, h) in enumerate(faces):
                    ## Getting Region of Interest for that Face
                    roi = image_array[y:y+h, x:x+w]

                    # initial database for known faces
                    database_data = initialize_data()

                    ## Getting Available information from Database
                    dataframe = database_data[COLS_INFO]

                    dataframe_new = dataframe.drop_duplicates(keep='first')
                    dataframe_new.reset_index(drop=True, inplace=True)

                    if dataframe_new.shape[0] > 0:
                        ## Getting Name of Visitor
                        name_visitor = dataframe_new.loc[0, 'Name']
                        attendance(visitor_id, name_visitor)

                        st.success(f"Welcome, {name_visitor}!")
                    else:
                        attendance(visitor_id, 'Unknown')
                        st.warning("Unknown person!")

            else:
                st.error('No human face detected.')

    if selected_menu == 'History':
        view_attendance()

    if selected_menu == 'Add to Database':
        password = st.text_input("Enter the password:", type="password")

        # Check if the password is correct
        if password == expected_password:
            col1, col2, col3 = st.columns(3)

            face_name = col1.text_input('Name:', '')
            pic_option = col2.radio('Upload Picture',
                                    options=["Upload a Picture",
                                             "Click a picture"])

            if pic_option == 'Upload a Picture':
                img_file_buffer = col3.file_uploader('Upload a Picture',
                                                     type=allowed_image_type)
                if img_file_buffer is not None:
                    # To read image file buffer with OpenCV:
                    file_bytes = np.asarray(bytearray(img_file_buffer.read()),
                                            dtype=np.uint8)

            elif pic_option == 'Click a picture':
                img_file_buffer = col3.camera_input("Click a picture")
                if img_file_buffer is not None:
                    # To read image file buffer with OpenCV:
                    file_bytes = np.frombuffer(img_file_buffer.getvalue(),
                                               np.uint8)

            if ((img_file_buffer is not None) & (len(face_name) > 1) &
                    st.button('Click to Save!')):
                # convert image from opened file to np.array
                image_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                with open(os.path.join(VISITOR_DB,
                                       f'{face_name}.jpg'), 'wb') as file:
                    file.write(img_file_buffer.getbuffer())
                    # st.success('Image Saved Successfully!')

                # Convert the image to grayscale
                gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

                # Load the Haar cascade files for face detection
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

                # Detect faces in the grayscale image
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                if len(faces) > 0:
                    encodesCurFrame = []

                    for (x, y, w, h) in faces:
                        roi = gray[y:y+h, x:x+w]
                        roi = cv2.resize(roi, (IMAGE_SIZE, IMAGE_SIZE))

                        encodesCurFrame.append(roi)

                    encodesCurFrame = np.array(encodesCurFrame)
                    encodesCurFrame = np.vstack(encodesCurFrame)

                    df_new = pd.DataFrame(encodesCurFrame, columns=[f"Encode{i+1}" for i in range(encodesCurFrame.shape[1])])

                    df_new[COLS_INFO] = face_name
                    df_new = df_new[COLS_INFO + COLS_ENCODE].copy()

                    # st.write(df_new)
                    # initial database for known faces
                    DB = initialize_data()
                    add_data_db(df_new)
        else:
            st.error("Incorrect password. Access denied.")

#######################################################
if __name__ == "__main__":
    main()
