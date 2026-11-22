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

VAD_THRESHOLD = 0.7 # minimum VAD probability to count a window as speech
MIN_SPEECH_RMS = 0.01 # minimum loudness (0-1 scale) to count a window as speech; filters distant voices
DEBUG_AUDIO_LEVELS = False # print loudness of speech-like windows to help tune MIN_SPEECH_RMS
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
LLM_MODEL = "claude-haiku-5-5"
LLM_MAX_TOKENS = 400 # caps reply length; includes thinking tokens
LLM_TIMEOUT_SECONDS = 30

# servo + speaker ESP32 (see ServoSpeakerControl.ino)
SERVO_SPEAKER_HOST = "servo-speaker.local"
SERVO_URL = f"http://{SERVO_SPEAKER_HOST}/servo"
SPEAKER_PORT = 81
SPEAKER_SAMPLE_RATE = 16_000 # must match SAMPLE_RATE in ServoSpeakerControl.ino
SPEAKER_CHUNK_BYTES = 4096 # bytes sent to the speaker per network write
SERVO_MIN_ANGLE = 0
SERVO_FALLBACK_SPEECH = "Okay." # spoken when Claude moves the servo without saying anything
SERVO_MAX_ANGLE = 180
ESP_TIMEOUT_SECONDS = 10
SPEAKER_ECHO_TAIL_SECONDS = 1.0 # keep ignoring the mic this long after speaking ends
