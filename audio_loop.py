import queue
import threading
import time

import numpy as np
import requests
import torch

from config import SAMPLE_RATE, SAMPLE_BYTES
from state_machine import StateMachine

# audio thread: stream -> VAD -> state machine -> utterance queue
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

            # mark when this window was received (same clock as the video frames)
            timestamp = time.monotonic()

            # feed audio through silero VAD model
            with torch.inference_mode():
                probability = vad(torch.from_numpy(samples), SAMPLE_RATE).item()

            # update state machine and update queue if we have a valid utterance
            utterance = sm.process(samples, probability, timestamp)
            if utterance is not None:
                try:
                    q.put_nowait(utterance)
                except queue.Full:
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass
