# interpreting streams/looping
import requests
import threading

# data 
import cv2 as cv
import numpy as np
from scipy.io.wavfile import write

# model
from faster_whisper import WhisperModel

# constants
SAMPLE_RATE = 16_000

# thread functions
def audio_loop(res: requests.Response, buffer: bytearray, event: threading.Event) -> None:
    for c in res.iter_content(chunk_size=2048):
        if event.is_set():
            break
        
        # extend buffer with this new chunk
        buffer.extend(c)

if __name__ == "__main__":
    # initialize whisper model
    model_size = "small" 
    model = WhisperModel(model_size, device="cpu", compute_type="float32")

    # get video feed from ESP32 server 
    cap = cv.VideoCapture("http://10.0.0.89/")
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # get audio feed from ESP32 server
    response = requests.get("http://10.0.0.89:81/", stream=True)

    # start audio thread
    buffer = bytearray()
    event = threading.Event()
    audio_thread = threading.Thread(target=audio_loop, args=(response, buffer, event,))
    audio_thread.start()

    # run main/video loop
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # convert audio buffer into samples
        complete_samples = len(buffer) // 4
        complete_bytes = complete_samples * 4 

        # get 5 seconds of audio
        if complete_samples >= 80_000:
            # assemble samples of audio
            samples = np.frombuffer(buffer[:SAMPLE_RATE * 5 * 4], dtype=np.int32) >> 8
            samples = samples.astype(np.float32) / (2 ** 23)
            
            # feed audio through whisper model
            segments, _ = model.transcribe(
                samples,
                language='en'
            )
            for s in segments:
                print(s.text)

            break

    
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

    # set event once video loop is finished
    event.set()

    # join thread
    audio_thread.join()

    # cleanup
    cap.release()
    cv.destroyAllWindows()
    response.close()