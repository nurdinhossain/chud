from collections import deque

import numpy as np

from config import VAD_THRESHOLD, MAX_UTTERANCE_WINDOWS

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
            if self.silent_periods >= 22 or len(self.recording) >= MAX_UTTERANCE_WINDOWS:
                self.mode = 0
                self.silent_periods = 0
                ret = np.concatenate(self.recording)
                self.recording.clear()
                self.pre_roll.clear()

                return ret

        return None
