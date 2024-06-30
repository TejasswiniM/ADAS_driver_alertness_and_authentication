import os
import numpy as np
import cv2
import pickle
from sface import SFace

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],
    [cv2.dnn.DNN_BACKEND_CUDA,   cv2.dnn.DNN_TARGET_CUDA],
    [cv2.dnn.DNN_BACKEND_CUDA,   cv2.dnn.DNN_TARGET_CUDA_FP16],
    [cv2.dnn.DNN_BACKEND_TIMVX,  cv2.dnn.DNN_TARGET_NPU],
    [cv2.dnn.DNN_BACKEND_CANN,   cv2.dnn.DNN_TARGET_NPU]
]

# current working directory
current_directory = os.path.dirname(os.path.abspath(__file__))

fr_model = os.path.join(current_directory, "..", "models", "face_recognition_sface_2021dec.onnx")
face_data_folder = os.path.join(current_directory, "..", "face_data", "vid_3", "aug_faces")
path = os.path.join(current_directory, "..", "face_data", "vid_3", "vid_3.pkl")

encoding_dict = {}

backend_id = backend_target_pairs[0][0]
target_id = backend_target_pairs[0][1]

# Instantiate SFace for face recognition
face_recognizer = SFace(modelPath=fr_model,
                   disType=0,
                   backendId=backend_id,
                   targetId=target_id)
                   
for subfolder in os.listdir(face_data_folder):
    #print("subfolder : ", subfolder)
    subfolder_path = os.path.join(face_data_folder, subfolder)
    if os.path.isdir(subfolder_path):
        #print("subfolder_path : ", subfolder_path)
        
        img_encode = []
        for file_name in os.listdir(subfolder_path):
            #print("file_name : ", file_name)
            if file_name.endswith(".jpg"):
                image_path = os.path.join(subfolder_path, file_name)

                img = cv2.imread(image_path)
               
                encode = face_recognizer.infer(img)
               
                img_encode.append(encode)

        encoding_dict[subfolder] = img_encode

with open(path, 'wb') as file:
    pickle.dump(encoding_dict, file)
   
print("Creation of face encodings completed.")