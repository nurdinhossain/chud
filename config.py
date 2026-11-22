# ESP32 streams
ESP32_IP = "10.0.0.89"
VIDEO_URL = f"http://{ESP32_IP}/"
AUDIO_URL = f"http://{ESP32_IP}:81/"

# audio constants
SAMPLE_RATE = 16_000
SECONDS = 32 / 1000 # 32 ms for Silero VAD model
BYTES_PER_HZ = 4
SAMPLES = int(SAMPLE_RATE * SECONDS)
SAMPLE_BYTES = SAMPLES * BYTES_PER_HZ

VAD_THRESHOLD = 0.5
MAX_UTTERANCE_SECONDS = 30
MAX_UTTERANCE_WINDOWS = int(MAX_UTTERANCE_SECONDS / SECONDS)

# video constants
FRAME_STORE_INTERVAL = 1.0 # seconds between retained video frames
FRAME_HISTORY_SIZE = 30 # keeps ~30 seconds of history
FRAME_MARGIN_SECONDS = 2.0 # extra history to include before an utterance starts
MAX_FRAMES_PER_UTTERANCE = 6 # caps images sent to the LLM (cost and latency)
FRAME_MAX_SIDE = 768 # frames are downscaled to this many pixels on the long side

# transcription
NO_SPEECH_THRESHOLD = 0.6 # drop whisper segments that are probably not speech

# LLM (needs the ANTHROPIC_API_KEY environment variable)
LLM_MODEL = "claude-sonnet-5-5"
LLM_MAX_TOKENS = 2048 # includes thinking tokens
LLM_TIMEOUT_SECONDS = 30
