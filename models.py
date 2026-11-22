from dataclasses import dataclass, field

import numpy as np

# a retained color frame and when we received it
@dataclass
class VideoFrame:
    timestamp: float # time.monotonic() when the frame was received
    frame: np.ndarray # original BGR color frame

# a finished chunk of speech and when it was spoken (not when it was transcribed)
@dataclass
class Utterance:
    samples: np.ndarray
    start_time: float # time.monotonic() of the first window, including pre-roll
    end_time: float # time.monotonic() of the last window
    frames: list[VideoFrame] = field(default_factory=list) # video from around the utterance

# a transcribed utterance, ready to send to the LLM
@dataclass
class Question:
    text: str
    frames: list[VideoFrame]
    end_time: float
