import queue
import threading
import time

import numpy as np
import requests
import torch

from config import SAMPLE_RATE, SAMPLE_BYTES, FRAME_MARGIN_SECONDS
from queue_utils import put_drop_oldest
from state_machine import StateMachine
from video import FrameHistory

# audio thread: stream -> VAD -> state machine -> (attach frames) -> utterance queue
def audio_loop(res: requests.Response, sm: StateMachine, vad, frame_history: FrameHistory, q: queue.Queue, speaking: threading.Event, event: threading.Event) -> None:
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

            # ignore the mic while the speaker is talking so we don't hear ourselves
            if speaking.is_set():
                sm.reset()
                continue

            # mark when this window was received (same clock as the video frames)
            timestamp = time.monotonic()

            # feed audio through silero VAD model
            with torch.inference_mode():
                probability = vad(torch.from_numpy(samples), SAMPLE_RATE).item()

            # update state machine; attach nearby video frames and queue any finished utterance
            utterance = sm.process(samples, probability, timestamp)
            if utterance is not None:
                utterance.frames = frame_history.get_range(utterance.start_time - FRAME_MARGIN_SECONDS, utterance.end_time)
                put_drop_oldest(q, utterance)
