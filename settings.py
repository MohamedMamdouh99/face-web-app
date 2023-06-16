import os
import pathlib
import streamlit as st
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import datetime

########################################################################################################################
# The Root Directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'

## We create a downloads directory within the streamlit static asset directory and we write output files to it
DOWNLOADS_PATH = (STREAMLIT_STATIC_PATH / "downloads")
if not DOWNLOADS_PATH.is_dir():
    DOWNLOADS_PATH.mkdir()

LOG_DIR = (STREAMLIT_STATIC_PATH / "logs")
if not LOG_DIR.is_dir():
    LOG_DIR.mkdir()

OUT_DIR = (STREAMLIT_STATIC_PATH / "output")
if not OUT_DIR.is_dir():
    OUT_DIR.mkdir()

VISITOR_DB = os.path.join(ROOT_DIR, "visitor_database")

if not os.path.exists(VISITOR_DB):
    os.mkdir(VISITOR_DB)

VISITOR_HISTORY = os.path.join(ROOT_DIR, "visitor_history")

if not os.path.exists(VISITOR_HISTORY):
    os.mkdir(VISITOR_HISTORY)

########################################################################################################################
## Defining Parameters

COLOR_DARK = (0, 0, 153)
COLOR_WHITE = (255, 255, 255)
COLS_INFO = ['Name']

## Database
data_path = VISITOR_DB
file_db = 'visitors_db.csv'  ## To store user information
file_history = 'visitors_history.csv'  ## To store visitor history information

## Image formats allowed
allowed_image_type = ['.png', 'jpg', '.jpeg']
################################################### Defining Function ##############################################
def initialize_data():
    if os.path.exists(os.path.join(data_path, file_db)):
        df = pd.read_csv(os.path.join(data_path, file_db))
    else:
        df = pd.DataFrame(columns=COLS_INFO)
        df.to_csv(os.path.join(data_path, file_db), index=False)

    return df

#################################################################
def add_data_db(df_visitor_details):
    try:
        df_all = pd.read_csv(os.path.join(data_path, file_db))

        if not df_all.empty:
            df_all = df_all.append(df_visitor_details, ignore_index=False)
            df_all.drop_duplicates(keep='first', inplace=True)
            df_all.reset_index(inplace=True, drop=True)
            df_all.to_csv(os.path.join(data_path, file_db), index=False)
            st.success('Details Added Successfully!')
        else:
            df_visitor_details.to_csv(os.path.join(data_path, file_db), index=False)
            st.success('Initiated Data Successfully!')

    except Exception as e:
        st.error(e)

#################################################################
def BGR_to_RGB(image_in_array):
    return cv2.cvtColor(image_in_array, cv2.COLOR_BGR2RGB)

#################################################################
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

#################################################################
def draw_bounding_boxes(image, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), COLOR_DARK, 2)
    return image

#################################################################
def process_image(image):
    faces = detect_faces(image)
    image_with_boxes = draw_bounding_boxes(image, faces)
    return image_with_boxes

#################################################################
def attendance(id, name):
    f_p = os.path.join(VISITOR_HISTORY, file_history)

    now = datetime.datetime.now()
    dtString = now.strftime('%Y-%m-%d %H:%M:%S')
    df_attendance_temp = pd.DataFrame(data={"id": [id], "visitor_name": [name], "Timing": [dtString]})

    if not os.path.isfile(f_p):
        df_attendance_temp.to_csv(f_p, index=False)
    else:
        df_attendance = pd.read_csv(f_p)
        df_attendance = df_attendance.append(df_attendance_temp)
        df_attendance.to_csv(f_p, index=False)

#################################################################
def view_attendance():
    f_p = os.path.join(VISITOR_HISTORY, file_history)
    df_attendance_temp = pd.DataFrame(columns=["id", "visitor_name", "Timing"])

    if not os.path.isfile(f_p):
        df_attendance_temp.to_csv(f_p, index=False)
    else:
        df_attendance_temp = pd.read_csv(f_p)

    df_attendance = df_attendance_temp.sort_values(by='Timing', ascending=False)
    df_attendance.reset_index(inplace=True, drop=True)

    st.write(df_attendance)

    if df_attendance.shape[0] > 0:
        id_chk = df_attendance.loc[0, 'id']
        id_name = df_attendance.loc[0, 'visitor_name']

        selected_img = st.selectbox('Search Image using ID', options=['None'] + list(df_attendance['id']))

        avail_files = [file for file in list(os.listdir(VISITOR_HISTORY))
                       if (file.endswith(tuple(allowed_image_type))) &
                       (file.startswith(selected_img) == True)]

        if len(avail_files) > 0:
            selected_img_path = os.path.join(VISITOR_HISTORY, avail_files[0])

            ## Displaying Image
            st.image(Image.open(selected_img_path))


########################################################################################################################
