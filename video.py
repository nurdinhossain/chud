import threading
from collections import deque

from config import MAX_FRAMES_PER_UTTERANCE
from models import VideoFrame

# rolling frame history: the main thread adds frames, other threads read snapshots
class FrameHistory:
    def __init__(self, maxlen: int):
        self._frames = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._frames)

    def add(self, frame: VideoFrame) -> None:
        with self._lock:
            self._frames.append(frame)

    # frames between start and end (newest few if too many). If none fall in
    # the interval, returns the latest frame from before it.
    def get_range(self, start: float, end: float) -> list[VideoFrame]:
        # hold the lock only to copy references, not while filtering
        with self._lock:
            snapshot = list(self._frames)

        selected = [f for f in snapshot if start <= f.timestamp <= end]
        if not selected:
            selected = [f for f in snapshot if f.timestamp < start][-1:]

        return selected[-MAX_FRAMES_PER_UTTERANCE:]
