# interpreting streams/looping
import requests
import threading
import anthropic
from dotenv import load_dotenv

# data
import cv2 as cv
import queue
import time

# model
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad

# project modules
from config import VIDEO_URL, AUDIO_URL, FRAME_STORE_INTERVAL, FRAME_HISTORY_SIZE, LLM_TIMEOUT_SECONDS
from models import VideoFrame
from state_machine import StateMachine
from audio_loop import audio_loop
from transcription import transcription_loop
from llm import llm_loop
from video import FrameHistory

# main function
def main():
    # load ANTHROPIC_API_KEY from .env
    load_dotenv()

    # initialize whisper model and state machine
    model_size = "small" 
    model = WhisperModel(model_size, device="cpu", compute_type="float32")
    vad = load_silero_vad()
    sm = StateMachine()
    utterance_queue = queue.Queue(maxsize=5)
    question_queue = queue.Queue(maxsize=3)
    client = anthropic.Anthropic(timeout=LLM_TIMEOUT_SECONDS) # reads ANTHROPIC_API_KEY

    # rolling history of color frames (main thread adds, audio thread reads snapshots)
    frame_history = FrameHistory(FRAME_HISTORY_SIZE)

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

    # start audio, transcription and LLM threads
    event = threading.Event()

    audio_thread = threading.Thread(target=audio_loop, args=(response, sm, vad, frame_history, utterance_queue, event,))
    transcript_thread = threading.Thread(target=transcription_loop, args=(utterance_queue, model, question_queue, event))
    llm_thread = threading.Thread(target=llm_loop, args=(question_queue, client, event))

    audio_thread.start()
    transcript_thread.start()
    llm_thread.start()

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
            frame_history.add(VideoFrame(timestamp=now, frame=frame))
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
    llm_thread.join()
    cap.release()
    cv.destroyAllWindows()
    response.close()

if __name__ == "__main__":
    main()