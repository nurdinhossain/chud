import queue
import threading

from faster_whisper import WhisperModel

from config import NO_SPEECH_THRESHOLD
from models import Question
from queue_utils import put_drop_oldest

# transcription thread: utterance queue -> whisper -> question queue
def transcription_loop(q: queue.Queue, model: WhisperModel, llm_q: queue.Queue, event: threading.Event) -> None:
    while not event.is_set():
        # feed audio through whisper model
        try:
            utt = q.get(timeout=0.1)
        except queue.Empty:
            continue

        segments, _ = model.transcribe(
            utt.samples,
            language='en'
        )

        # join segments into one string, skipping ones that are probably not speech
        text = " ".join(s.text.strip() for s in segments if s.no_speech_prob < NO_SPEECH_THRESHOLD).strip()
        if not text:
            continue

        print(f"[{utt.start_time:.2f}s - {utt.end_time:.2f}s] {text} ({len(utt.frames)} frames)")
        put_drop_oldest(llm_q, Question(text=text, frames=utt.frames, end_time=utt.end_time))
