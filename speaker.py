import os
import socket
import subprocess
import tempfile
import wave

from config import (
    SERVO_SPEAKER_HOST, SPEAKER_PORT, SPEAKER_SAMPLE_RATE, SPEAKER_CHUNK_BYTES, ESP_TIMEOUT_SECONDS,
)

# turn text into raw 16-bit mono PCM using the macOS `say` command
def synthesize(text: str) -> bytes:
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "speech.wav")
        subprocess.run(
            ["say", "-o", path, "--file-format=WAVE", f"--data-format=LEI16@{SPEAKER_SAMPLE_RATE}",
             "--channels=1", "-f", "-"], # text comes from stdin so it can't be read as an option
            input=text.encode("utf-8"), check=True,
        )
        with wave.open(path) as wav:
            return wav.readframes(wav.getnframes())

# stream audio to the speaker ESP32; it plays as the data arrives, so this takes about as long as the speech
def send_audio(pcm: bytes) -> None:
    with socket.create_connection((SERVO_SPEAKER_HOST, SPEAKER_PORT), timeout=ESP_TIMEOUT_SECONDS) as sock:
        for i in range(0, len(pcm), SPEAKER_CHUNK_BYTES):
            sock.sendall(pcm[i:i + SPEAKER_CHUNK_BYTES])

# say text out loud through the speaker
def speak(text: str) -> None:
    try:
        send_audio(synthesize(text))
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"Speaking failed: {e}")
