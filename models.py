from dataclasses import dataclass

import numpy as np

# a retained color frame and when we received it
@dataclass
class VideoFrame:
    timestamp: float # time.monotonic() when the frame was received
    frame: np.ndarray # original BGR color frame
