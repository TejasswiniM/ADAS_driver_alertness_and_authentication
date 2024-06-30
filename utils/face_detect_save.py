#Import libraries of Python
import argparse
import numpy as np
import cv2
from yunet import YuNet
import os

# current working directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Save cropped faces folder path
save_crop_face_path = os.path.join(current_directory, "..", "face_data", "vid_2", "cropped_faces")

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
        
		
def crop_save_faces(video_path, face_det_model_path):

    face_detector = FACE_DETECTION(face_det_model_path)

    cap = cv2.VideoCapture(video_path)

    frame_no = 1
    while True:
        ret, frame = cap.read()
       
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
       
        crop_faces = []
        
        #Face detection and recognition
        face_coords = face_detector.detect_faces(frame)
        for face_bbox in face_coords:
            face = frame[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]]
            crop_faces.append(face)
            #cv2.rectangle(frame, (face_bbox[0], face_bbox[1]), (face_bbox[2], face_bbox[3]), (0, 0, 255), 3)   
            
        cv2.imshow("interior_media_execution", frame)

        output_path = os.path.join(save_crop_face_path, f'frame_{frame_no}.jpg')
        for cropped in crop_faces:
            #cv2.imshow("crop_face", cropped)
            cv2.imwrite(output_path, cropped)

        frame_no += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
       
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save detected cropped faces")
    parser.add_argument("--input-media", type=str, default= 0, help="Path to video source")
    parser.add_argument("--face-det-model", type=str, default=r"..\models\face_detection_yunet_2023mar.onnx", help="Path to Face detection model")
    args = parser.parse_args()
   
    if args.input_media != 0:
        media_path = os.path.join(current_directory, args.input_media)
    else:
        media_path = args.input_media
    
    face_detection_model = os.path.join(current_directory, args.face_det_model)
    
    crop_save_faces(media_path, face_detection_model)