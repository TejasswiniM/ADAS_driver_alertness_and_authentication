#Import libraries of Python
import cv2
import numpy as np
import argparse
import os
import mediapipe as mp
import onnxruntime
from utils.yunet import YuNet
from utils.sface import SFace
import pickle
from scipy.spatial.distance import cosine
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import csv


# current working directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Encoded face data
#encodings_path = os.path.join(current_directory, "face_data", "vid_1_1280", "vid_1_1280.pkl")
encodings_path = os.path.join(current_directory, "face_data", "vid_2", "vid_2.pkl")
#encodings_path = os.path.join(current_directory, "face_data", "vid_3", "vid_3.pkl")

with open(encodings_path, 'rb') as f:
    encoding_dict = pickle.load(f)

# Music file initialize and loading
mixer.init()
mixer.music.load(os.path.join(current_directory, "sample_videos", "music.wav"))

# CSV file path
csv_file_path = os.path.join(current_directory, "EDA", "drowsiness_info.csv")

#Face detection class
class FACE_DETECTION:
    def __init__(self, model_path):
        self.face_detection_model = model_path
        self.backend_target_pairs = [
            [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],
            [cv2.dnn.DNN_BACKEND_CUDA,   cv2.dnn.DNN_TARGET_CUDA],
            [cv2.dnn.DNN_BACKEND_CUDA,   cv2.dnn.DNN_TARGET_CUDA_FP16],
            [cv2.dnn.DNN_BACKEND_TIMVX,  cv2.dnn.DNN_TARGET_NPU],
            [cv2.dnn.DNN_BACKEND_CANN,   cv2.dnn.DNN_TARGET_NPU]
        ]

        self.backend_id = self.backend_target_pairs[0][0]
        self.target_id = self.backend_target_pairs[0][1]
        self.face_input_size = (320, 320)
        self.conf_threshold = 0.9
        self.nms_threshold = 0.3
        self.top_k = 5000
        self.model = YuNet(modelPath=self.face_detection_model,
                  inputSize=self.face_input_size,
                  confThreshold=self.conf_threshold,
                  nmsThreshold=self.nms_threshold,
                  topK=self.top_k,
                  backendId=self.backend_id,
                  targetId=self.target_id)
    
    def detect_faces(self, frame):
        h, w, _ = frame.shape
        h_ratio, w_ratio = frame.shape[0]/self.face_input_size[1], frame.shape[1]/self.face_input_size[0]

        # Inference
        self.model.setInputSize([w, h])
        results = self.model.infer(frame)

        face_cords = []
        for det in (results if results is not None else []):
            bbox = det[0:4].astype(np.int32)
            face_cord = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
            #face_cord = [int(bbox[0]*w_ratio), int(bbox[1]*h_ratio), int((bbox[0]+bbox[2])*w_ratio), int((bbox[1]+bbox[3])*h_ratio)]
            face_cords.append(face_cord)

        return face_cords


#Face recognition class
class FACE_RECOGNITION:
    def __init__(self, face_rec_model):
        self.face_recognition_model = face_rec_model
        self.backend_target_pairs = [
            [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],
            [cv2.dnn.DNN_BACKEND_CUDA,   cv2.dnn.DNN_TARGET_CUDA],
            [cv2.dnn.DNN_BACKEND_CUDA,   cv2.dnn.DNN_TARGET_CUDA_FP16],
            [cv2.dnn.DNN_BACKEND_TIMVX,  cv2.dnn.DNN_TARGET_NPU],
            [cv2.dnn.DNN_BACKEND_CANN,   cv2.dnn.DNN_TARGET_NPU]
        ]

        self.backend_id = self.backend_target_pairs[0][0]
        self.target_id = self.backend_target_pairs[0][1]

        self.recognition_threshold = 0.5
        self.face_recognizer = SFace(modelPath=self.face_recognition_model, disType=0, backendId=self.backend_id, targetId=self.target_id)
   
    def recognise_faces(self, face):
        face_rec_result = {}
        face_feature = self.face_recognizer.infer(face)
        unknown_key = 'unknown'
        min_dict = {}            
        for key, val in encoding_dict.items():
            face_encode_list = encoding_dict.get(key)
            min_dist = float("inf")    
            for i in range(len(face_encode_list)):
                dist = cosine(face_feature[0], face_encode_list[i][0])
                if dist < min_dist and dist < self.recognition_threshold:
                    min_dist = dist
           
            if min_dist == float("inf"):
                min_dict[unknown_key] = min_dist
           
            else:
                min_dict[key] = min_dist
               
        best_key = min(min_dict, key=min_dict.get)
        best_min_value = min_dict[best_key]
        
        face_rec_result[best_key] = best_min_value
        
        for key, value in face_rec_result.items():
            if value == float("inf"):
                return key
            else:
                return key


#Drowsiness detection class
class DROWSINESS_DETECTION:
    # change consecutive_frames_threshold and ear_threshold as per requirements
    def __init__(self, onnx_model_path, consecutive_frames_threshold=7, ear_threshold=0.3): 
        # Load the ONNX model
        self.sess = onnxruntime.InferenceSession(onnx_model_path)

        # Initialize variables for drowsiness detection
        self.consecutive_frames_threshold = consecutive_frames_threshold
        self.ear_threshold = ear_threshold
        self.consecutive_frames = 0 

        # Open CSV file for writing
        self.csv_file = open(csv_file_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Frame Number", "Left Eye EAR", "Right Eye EAR", "Average EAR", "Eye State"])

    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])

        ear = (A + B) / (2.0 * C)
        return ear

    def detect_drowsiness(self, frame, frame_number, face_landmarks):
        left_eye_landmarks = [face_landmarks.landmark[i] for i in range(159, 145, -1)]
        right_eye_landmarks = [face_landmarks.landmark[i] for i in range(386, 380, -1)]

        h, w, _ = frame.shape
        left_eye_coords = np.array([(lm.x * w, lm.y * h) for lm in left_eye_landmarks], dtype=int)
        right_eye_coords = np.array([(lm.x * w, lm.y * h) for lm in right_eye_landmarks], dtype=int)
 
        # Extract left and right eye ROIs
        left_eye_roi = frame[min(left_eye_coords[:, 1]):max(left_eye_coords[:, 1]), min(left_eye_coords[:, 0]):max(left_eye_coords[:, 0])]
        right_eye_roi = frame[min(right_eye_coords[:, 1]):max(right_eye_coords[:, 1]), min(right_eye_coords[:, 0]):max(right_eye_coords[:, 0])]
        
        #print("left_eye_roi : ", left_eye_roi.shape)
        #print("right_eye_roi : ", right_eye_roi.shape)
        
        # Calculate the mid-point of the eye ROI height
        left_eye_midpoint = (min(left_eye_coords[:, 1]) + max(left_eye_coords[:, 1])) // 2
        right_eye_midpoint = (min(right_eye_coords[:, 1]) + max(right_eye_coords[:, 1])) // 2

        # Crop only the upper half of the eye ROIs
        upper_half_left_eye_roi = frame[min(left_eye_coords[:, 1]):left_eye_midpoint, min(left_eye_coords[:, 0]):max(left_eye_coords[:, 0])]
        upper_half_right_eye_roi = frame[min(right_eye_coords[:, 1]):right_eye_midpoint, min(right_eye_coords[:, 0]):max(right_eye_coords[:, 0])]

        #cv2.imshow("upper_half_left_eye_roi : ", upper_half_left_eye_roi)
        #cv2.imshow("upper_half_right_eye_roi : ", upper_half_right_eye_roi)

        # Resize the ROIs to the required input size for the ONNX model
        left_eye_roi_resized = cv2.resize(upper_half_left_eye_roi, (224, 224))
        right_eye_roi_resized = cv2.resize(upper_half_right_eye_roi, (224, 224))

        # Convert to float32 and add batch dimension
        left_eye_input = np.expand_dims(left_eye_roi_resized.astype(np.float32), axis=0)
        right_eye_input = np.expand_dims(right_eye_roi_resized.astype(np.float32), axis=0)

        # Perform inference for left eye
        left_eye_result = self.sess.run([self.sess.get_outputs()[0].name], {self.sess.get_inputs()[0].name: left_eye_input})

        # Perform inference for right eye
        right_eye_result = self.sess.run([self.sess.get_outputs()[0].name], {self.sess.get_inputs()[0].name: right_eye_input})

        # Combine results from left and right eyes
        ear_left = self.eye_aspect_ratio(left_eye_coords)
        ear_right = self.eye_aspect_ratio(right_eye_coords)
        ear_avg = (ear_left + ear_right) / 2.0

        prediction = (left_eye_result[0][0][0] + right_eye_result[0][0][0]) / 2.0
        #print("prediction : ", prediction)

        if ear_avg < self.ear_threshold:
            # Check for consecutive closed eye frames
            if self.consecutive_frames >= self.consecutive_frames_threshold:
                cv2.putText(frame, "****!ALERT!****", (int(w/4), int(9*h/10)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                mixer.music.play()
                eye_state = "Closed Eyes"
                alert_color = (0, 0, 255)  # Red for Eyes Closed
            else:
                eye_state = "Opened Eyes"
                alert_color = (0, 255, 0)  # Green for Eyes Opened
            self.consecutive_frames += 1
        else:
            self.consecutive_frames = 0
            eye_state = "Opened Eyes"
            alert_color = (0, 255, 0)  # Green for Eyes Opened

        # Write data to CSV
        self.csv_writer.writerow([frame_number, ear_left, ear_right, ear_avg, eye_state])
        self.csv_file.flush()

        # Display eye state on the frame
        cv2.rectangle(frame, (10, 40), (160, 80), (0, 0, 0), -1)  # Black rectangle
        cv2.putText(frame, f'{eye_state}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2)

        cv2.putText(frame, f'EAR: {ear_avg:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def close(self):
        # Release resources and close CSV file
        self.csv_file.close()
        mixer.quit()


#Main execution function
def driver_alert_auth_prog_infer(video_path, face_det_model_path, face_rec_model_path, drowiness_det_model_path):

    face_detector = FACE_DETECTION(face_det_model_path)
    face_recognition = FACE_RECOGNITION(face_rec_model_path)
    drowsiness_detector = DROWSINESS_DETECTION(drowiness_det_model_path)
    
    # Initialize FaceMesh
    mp_face_mesh = mp.solutions.face_mesh

    # Set up FaceMesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.2, min_tracking_confidence=0.2)

    cap = cv2.VideoCapture(video_path)
    frame_num = 1
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        face_detect_rec_input = frame.copy()
        drowiness_detect_input = frame.copy()
        
        cropped_faces = []
        cropped_faces_bbox = []
        
        #Face detection
        face_coords = face_detector.detect_faces(face_detect_rec_input)
        for face_bbox in face_coords:
            face = face_detect_rec_input[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]]
            cropped_faces.append(face)
            cropped_faces_bbox.append(face_bbox)
            #cv2.rectangle(face_detect_rec_input, (face_bbox[0], face_bbox[1]), (face_bbox[2], face_bbox[3]), (0, 0, 255), 2)   
            
        #Face recognition
        for face, face_box in zip (cropped_faces, cropped_faces_bbox):
            fr_result = face_recognition.recognise_faces(face)
            
            if len(fr_result) > 0:
                if fr_result == 'unknown':
                    result_image = cv2.rectangle(face_detect_rec_input, (face_box[0], face_box[1]), (face_box[2], face_box[3]), (0, 0, 255), 2)
                    result_image = cv2.putText(result_image, fr_result, (face_box[0], face_box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                else:
                    result_image = cv2.rectangle(face_detect_rec_input, (face_box[0], face_box[1]), (face_box[2], face_box[3]), (0, 255, 0), 2)
                    result_image = cv2.putText(result_image, fr_result, ((face_box[0], face_box[1])[0],(face_box[0], face_box[1])[1] - 5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,200,200), 2)

        cv2.imshow("Face_detect_recognise", face_detect_rec_input)

        #Drowsiness Detection
        rgb_frame = cv2.cvtColor(drowiness_detect_input, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                drowsiness_detector.detect_drowsiness(drowiness_detect_input, frame_num, face_landmarks)

        cv2.imshow("Drowsiness_detect", drowiness_detect_input)
        

        frame_num +=1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
       
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADAS driver alertness and authentication process")
    parser.add_argument("--input-media", type=str, default= 0, help="Path to video source")
    parser.add_argument("--face-det-model", type=str, default=r"models\face_detection_yunet_2023mar.onnx", help="Path to Face detection model")
    parser.add_argument("--face-rec-model", type=str, default=r"models\face_recognition_sface_2021dec.onnx", help="Path to Face recognition model")
    parser.add_argument("--drowsiness-det-model", type=str, default=r"models\drowsiness_detection_model.onnx", help="Path to Drowsiness detect model")
    args = parser.parse_args()

    if args.input_media != 0:
        media_path = os.path.join(current_directory, args.input_media)
    else:
        media_path = args.input_media
    
    face_detection_model = os.path.join(current_directory, args.face_det_model)
    face_recognition_model = os.path.join(current_directory, args.face_rec_model)
    drowsiness_detection_model = os.path.join(current_directory, args.drowsiness_det_model)

    driver_alert_auth_prog_infer(media_path, face_detection_model, face_recognition_model, drowsiness_detection_model)
