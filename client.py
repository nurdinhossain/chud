import requests
import numpy as np
import cv2 as cv
import time

if __name__ == "__main__":
    # get video feed from ESP32 server 
    cap = cv.VideoCapture("http://10.0.0.89/")
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # infinite loop to grab camera frames
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Our operations on the frame come here
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Display the resulting frame
        cv.imshow('frame', gray)
        if cv.waitKey(1) == ord('q'):
            break
    
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()