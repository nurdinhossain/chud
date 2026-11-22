from collections import deque

import numpy as np

from config import VAD_THRESHOLD, MAX_UTTERANCE_WINDOWS
from models import Utterance

# audio state machine
class StateMachine:
    def __init__(self):
        self.mode = 0 # 0 = listening, 1 = recording
        self.pre_roll = deque(maxlen=10) # (timestamp, samples) pairs
        self.recording = [] # stores the recorded conversation
        self.start_time = 0.0 # timestamp of the first window in the recording
        self.silent_periods = 0

    def process(self, samples: np.ndarray, p: float, timestamp: float) -> Utterance | None:
        if self.mode == 0:
            if p < VAD_THRESHOLD:
                self.pre_roll.append((timestamp, samples))
            else:
                self.mode = 1

                # the recording starts at the oldest pre-roll window (if any)
                self.start_time = self.pre_roll[0][0] if self.pre_roll else timestamp
                self.recording.extend(window for _, window in self.pre_roll)
                self.recording.append(samples)
        else:
            self.recording.append(samples)

            if p < VAD_THRESHOLD:
                self.silent_periods += 1
            else:
                self.silent_periods = 0

            # reset upon long silence (conversation assumed to have ended)
            if self.silent_periods >= 22 or len(self.recording) >= MAX_UTTERANCE_WINDOWS:
                self.mode = 0
                self.silent_periods = 0
                ret = Utterance(
                    samples=np.concatenate(self.recording),
                    start_time=self.start_time,
                    end_time=timestamp
                )
                self.recording.clear()
                self.pre_roll.clear()

                return ret

        return None
