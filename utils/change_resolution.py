import time
import cv2

def rescale_frame(frame_input, percent=75):
    width = int(frame_input.shape[1] * percent / 100)
    height = int(frame_input.shape[0] * percent / 100)
    dim = (1280,720)
    return cv2.resize(frame_input, dim, interpolation=cv2.INTER_AREA)

#cap = cv2.VideoCapture('vid_4.mp4')
#cap = cv2.VideoCapture('pexels-mario-angel-5915075-1920x1080-30fps.mp4')
#cap = cv2.VideoCapture('Dscn0545.m4v')
cap = cv2.VideoCapture(r"C:\Users\Admin\Desktop\code\sample_videos\vid_1.mp4")

if cap.isOpened():
    ret, frame = cap.read()
    rescaled_frame = rescale_frame(frame)
    (h, w) = rescaled_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(r"C:\Users\Admin\Desktop\code\sample_videos\vid_1_1280.mp4", fourcc, 15.0, (w, h), True)

else:
    print("Camera is not opened")

while cap.isOpened():
    ret, frame = cap.read()
    rescaled_frame = rescale_frame(frame)
    # write the output frame to file
    writer.write(rescaled_frame)
    cv2.imshow("Output", rescaled_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
writer.release()