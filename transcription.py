import queue
import threading

from faster_whisper import WhisperModel

# transcription thread: utterance queue -> whisper -> printed text
def transcription_loop(q: queue.Queue, model: WhisperModel, event: threading.Event) -> None:
    while not event.is_set():
        # feed audio through whisper model
        try:
            utt = q.get(timeout=0.1)
        except queue.Empty:
            continue

        segments, _ = model.transcribe(
            utt,
            language='en'
        )

        for s in segments:
            print(s.text)
