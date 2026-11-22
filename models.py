from dataclasses import dataclass

import numpy as np

# a finished chunk of speech and when it was spoken (not when it was transcribed)
@dataclass
class Utterance:
    samples: np.ndarray
    start_time: float # time.monotonic() of the first window, including pre-roll
    end_time: float # time.monotonic() of the last window

# a retained color frame and when we received it
@dataclass
class VideoFrame:
    timestamp: float # time.monotonic() when the frame was received
    frame: np.ndarray # original BGR color frame
