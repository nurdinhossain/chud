# interpreting streams/looping
import requests
import threading

# data
import cv2 as cv
from collections import deque
import queue
import time

# model
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad

# project modules
from config import VIDEO_URL, AUDIO_URL, FRAME_STORE_INTERVAL, FRAME_HISTORY_SIZE
from models import VideoFrame
from state_machine import StateMachine
from audio_loop import audio_loop
from transcription import transcription_loop

# main function
def main():
    # initialize whisper model and state machine
    model_size = "small" 
    model = WhisperModel(model_size, device="cpu", compute_type="float32")
    vad = load_silero_vad()
    sm = StateMachine()
    utterance_queue = queue.Queue(maxsize=5)

    # get video feed from ESP32 server 
    cap = cv.VideoCapture(VIDEO_URL)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # get audio feed from ESP32 server
    try:
        response = requests.get(AUDIO_URL, stream=True, timeout=5)
    except requests.exceptions.Timeout:
        print("Attempt to connect to audio feed timed out. Exiting script.")
        exit()

    # start audio and transcription thread
    event = threading.Event()

    audio_thread = threading.Thread(target=audio_loop, args=(response, sm, vad, utterance_queue, event,))
    transcript_thread = threading.Thread(target=transcription_loop, args=(utterance_queue, model, event))

    audio_thread.start()
    transcript_thread.start()

    # rolling history of color frames (owned by the main thread, so no lock needed yet)
    frame_history = deque(maxlen=FRAME_HISTORY_SIZE)
    last_stored_time = float("-inf")

    # run main/video loop
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # store one color frame per interval
        now = time.monotonic()
        if now - last_stored_time >= FRAME_STORE_INTERVAL:
            frame_history.append(VideoFrame(timestamp=now, frame=frame))
            last_stored_time = now
            print(f"Stored frame at {now:.2f}s | history size: {len(frame_history)}")

        # Our operations on the frame come here
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Display the resulting frame
        cv.imshow('frame', gray)
        if cv.waitKey(1) == ord('q'):
            break

    # set event once video loop is finished
    event.set()

    # cleanup
    audio_thread.join()
    transcript_thread.join()
    cap.release()
    cv.destroyAllWindows()
    response.close()

if __name__ == "__main__":
    main()