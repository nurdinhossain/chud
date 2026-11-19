# interpreting streams/looping
import requests
import threading

# data 
import cv2 as cv
import numpy as np
from collections import deque
import queue

# model
import torch
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad

# constants
SAMPLE_RATE = 16_000
SECONDS = 32 / 1000 # 32 ms for Silero VAD model
BYTES_PER_HZ = 4
SAMPLES = int(SAMPLE_RATE * SECONDS)
SAMPLE_BYTES = SAMPLES * BYTES_PER_HZ

VAD_THRESHOLD = 0.5

# audio state machine
class StateMachine:
    def __init__(self):
        self.mode = 0 # 0 = listening, 1 = recording
        self.pre_roll = deque(maxlen=10)
        self.recording = [] # stores the recorded conversation
        self.silent_periods = 0

    def process(self, samples: np.ndarray, p: float) -> np.ndarray | None:
        if self.mode == 0:
            if p < VAD_THRESHOLD:
                self.pre_roll.append(samples)
            else:
                self.mode = 1
                self.recording.extend(self.pre_roll)
                self.recording.append(samples)
        else:
            self.recording.append(samples)

            if p < VAD_THRESHOLD:
                self.silent_periods += 1
            else:
                self.silent_periods = 0

            # reset upon long silence (conversation assumed to have ended)
            if self.silent_periods >= 22:
                self.mode = 0
                self.silent_periods = 0
                ret = np.concatenate(self.recording)
                self.recording.clear()
                self.pre_roll.clear()

                return ret
            
        return None

# thread functions
def audio_loop(res: requests.Response, sm: StateMachine, vad, q: queue.Queue, event: threading.Event) -> None:
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

            # update state machine and update queue if we have a valid utterance
            utterance = sm.process(samples, probability)
            if utterance is not None:
                q.put(utterance)

def transcription_loop(q: queue.Queue, model: WhisperModel, event: threading.Event) -> None:
    while not event.is_set():
        # feed audio through whisper model
        try:
            utt = q.get(timeout=0.1)
        except queue.Empty:
            continue

        segments, _ = model.transcribe(
            utt,
            language='en'
        )

        for s in segments:
            print(s.text)

# main function
def main():
    # initialize whisper model and state machine
    model_size = "small" 
    model = WhisperModel(model_size, device="cpu", compute_type="float32")
    vad = load_silero_vad()
    sm = StateMachine()
    utterance_queue = queue.Queue()

    # get video feed from ESP32 server 
    cap = cv.VideoCapture("http://10.0.0.89/")
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # get audio feed from ESP32 server
    try:
        response = requests.get("http://10.0.0.89:81/", stream=True, timeout=5)
    except requests.exceptions.Timeout:
        print("Attempt to connect to audio feed timed out. Exiting script.")
        exit()

    # start audio and transcription thread
    event = threading.Event()

    audio_thread = threading.Thread(target=audio_loop, args=(response, sm, vad, utterance_queue, event,))
    transcript_thread = threading.Thread(target=transcription_loop, args=(utterance_queue, model, event))

    audio_thread.start()
    transcript_thread.start()

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
    transcript_thread.join()
    cap.release()
    cv.destroyAllWindows()
    response.close()

if __name__ == "__main__":
    main()