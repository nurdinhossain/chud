import requests
import numpy as np
import time
import threading
import cv2 as cv

# thread functions
def video_loop(cap):
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

def audio_loop(res, event):
    for c in res.iter_content(chunk_size=2048):
        if event.is_set():
            break
        print(np.frombuffer(c, dtype=np.int32))

if __name__ == "__main__":
    # get video feed from ESP32 server 
    cap = cv.VideoCapture("http://10.0.0.89/")
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # get audio feed from ESP32 server
    response = requests.get("http://10.0.0.89:81/", stream=True)

    # start audio thread
    event = threading.Event()
    audio_thread = threading.Thread(target=audio_loop, args=(response, event))
    audio_thread.start()

    # run video loop
    video_loop(cap)

    # set event once video loop is finished
    event.set()

    # join thread
    audio_thread.join()

    # cleanup
    cap.release()
    cv.destroyAllWindows()
    response.close()