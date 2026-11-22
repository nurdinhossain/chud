import base64
import queue
import threading

import anthropic
import cv2 as cv
import numpy as np

from config import LLM_MODEL, LLM_MAX_TOKENS, FRAME_MAX_SIDE
from models import Question

SYSTEM_PROMPT = (
    "You are a voice assistant built into a camera-and-microphone device. "
    "The user speaks to you, and you are shown frames from their camera in time order, "
    "each labeled with how long before they finished speaking it was taken. "
    "Use the frames to answer questions about what they were looking at or holding. "
    "Answer in one or two short sentences, since the reply may be read aloud."
)

# downscale a frame and convert it to a base64 JPEG image block
def encode_frame(frame: np.ndarray) -> dict:
    scale = FRAME_MAX_SIDE / max(frame.shape[:2])
    if scale < 1:
        frame = cv.resize(frame, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)

    _, jpeg = cv.imencode(".jpg", frame)
    data = base64.standard_b64encode(jpeg.tobytes()).decode("utf-8")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}

# frames first (each labeled with its age), then what the user said
def build_content(question: Question) -> list[dict]:
    content = []
    for f in question.frames:
        age = question.end_time - f.timestamp
        content.append({"type": "text", "text": f"Frame from {age:.1f}s before the user finished speaking:"})
        content.append(encode_frame(f.frame))

    content.append({"type": "text", "text": f'The user said: "{question.text}"'})
    return content

# send the question to Claude and print the reply as it streams in
def ask_claude(client: anthropic.Anthropic, question: Question) -> None:
    with client.messages.stream(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        output_config={"effort": "low"}, # fast replies; raise for harder questions
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_content(question)}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
        final = stream.get_final_message()

    print()
    if final.stop_reason == "refusal":
        print("(Claude declined to answer)")

# LLM thread: question queue -> Claude -> printed reply
def llm_loop(q: queue.Queue, client: anthropic.Anthropic, event: threading.Event) -> None:
    while not event.is_set():
        try:
            question = q.get(timeout=0.1)
        except queue.Empty:
            continue

        try:
            ask_claude(client, question)
        except anthropic.APIError as e: # the SDK already retries transient errors
            print(f"\nClaude request failed: {e}")
