from collections import deque

import numpy as np

from config import VAD_THRESHOLD, MIN_SPEECH_RMS, DEBUG_AUDIO_LEVELS, MAX_UTTERANCE_WINDOWS
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
        # a window only counts as speech if VAD hears speech AND it is loud enough
        # (background voices pass VAD but are quieter than someone near the mic)
        rms = float(np.sqrt(np.mean(samples ** 2)))
        if DEBUG_AUDIO_LEVELS and p >= VAD_THRESHOLD:
            print(f"speech-like window: rms={rms:.4f} (min {MIN_SPEECH_RMS}) vad={p:.2f}")
        is_speech = p >= VAD_THRESHOLD and rms >= MIN_SPEECH_RMS

        if self.mode == 0:
            if not is_speech:
                self.pre_roll.append((timestamp, samples))
            else:
                self.mode = 1

                # the recording starts at the oldest pre-roll window (if any)
                self.start_time = self.pre_roll[0][0] if self.pre_roll else timestamp
                self.recording.extend(window for _, window in self.pre_roll)
                self.recording.append(samples)
        else:
            self.recording.append(samples)

            if is_speech:
                self.silent_periods = 0
            else:
                self.silent_periods += 1

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
