# interpreting streams/looping
import requests
import threading

# data 
import cv2 as cv
import numpy as np
from collections import deque
from scipy.io.wavfile import write

# model
import torch
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, get_speech_timestamps

# constants
SAMPLE_RATE = 16_000
SECONDS = 32 / 1000 # 32 ms for Silero VAD model
BYTES_PER_HZ = 4
SAMPLES = int(SAMPLE_RATE * SECONDS)
SAMPLE_BYTES = SAMPLES * BYTES_PER_HZ

# thread functions
def audio_loop(res: requests.Response, event: threading.Event) -> None:
    buffer = bytearray()
    for c in res.iter_content(chunk_size=2048):
        if event.is_set():
            break
        
        # extend buffer with this new chunk
        buffer.extend(c)
        
        # convert audio buffer into samples
        while len(buffer) >= SAMPLE_BYTES:
            samples = np.frombuffer(buffer[:SAMPLE_BYTES], dtype=np.int32) >> 8
            samples = samples.astype(np.float32) / (2 ** 23)

            # delete sample so we don't read it again
            del buffer[:SAMPLE_BYTES]
            
            # feed audio through silero VAD model
            with torch.inference_mode():
                probability = vad(torch.from_numpy(samples), SAMPLE_RATE).item()
            print(probability)

            # feed audio through whisper model
            """segments, _ = model.transcribe(
                samples,
                language='en'
            )

            for s in segments:
                print(s.text)

            break"""

# audio state machine
class StateMachine:
    def __init__(self):
        self.mode = 0 # 0 = listening, 1 = recording

if __name__ == "__main__":
    # initialize whisper model, pre-roll, and state machine
    model_size = "small" 
    model = WhisperModel(model_size, device="cpu", compute_type="float32")
    vad = load_silero_vad()
    pre_roll = deque(maxlen=10)
    sm = StateMachine()

    # get video feed from ESP32 server 
    cap = cv.VideoCapture("http://10.0.0.89/")
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # get audio feed from ESP32 server
    response = requests.get("http://10.0.0.89:81/", stream=True)

    # start audio thread
    event = threading.Event()
    audio_thread = threading.Thread(target=audio_loop, args=(response, event,))
    audio_thread.start()

    # run main/video loop
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

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
    cap.release()
    cv.destroyAllWindows()
    response.close()