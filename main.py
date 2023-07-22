import cv2
import imutils
import time

cv2.namedWindow("Image Profile")
camera = cv2.VideoCapture(0)

while True:
    start_time = time.time()
    ret, frame = camera.read()
    frame = imutils.resize(frame, width=720)

    end_time = time.time() - start_time    
    FPS = 1/end_time

    cv2.putText(frame,
                f"FPS: {round(FPS,3)}",
                (10,50),
                cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),
                2)
    cv2.imshow('Face info',frame)

    if cv2.waitKey(1) &0xFF == ord('q'):
        break