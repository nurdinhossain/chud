import base64
import queue
import threading
import time

import anthropic
import cv2 as cv
import numpy as np

from config import LLM_MODEL, LLM_MAX_TOKENS, FRAME_MAX_SIDE, SPEAKER_ECHO_TAIL_SECONDS, SERVO_FALLBACK_SPEECH
from models import Question
from servo import move_servo
from speaker import speak

SYSTEM_PROMPT = (
    "You are a voice assistant built into a camera-and-microphone device. "
    "The user speaks to you, and you are shown frames from their camera in time order, "
    "each labeled with how long before they finished speaking it was taken. "
    "Use the frames to answer questions about what they were looking at or holding. "
    "Answer in one or two short sentences, since the reply is spoken aloud through a speaker. "
    "You can also turn a servo with the move_servo tool: 0 is far left, 90 is center, 180 is far right. "
    "Use it when the user asks you to move, turn or point. Every reply must include a short spoken sentence, "
    "even when you call the tool (for example, \"Turning right.\")."
)

MOVE_SERVO_TOOL = {
    "name": "move_servo",
    "description": "Turn the servo to an angle in degrees, from 0 (far left) to 180 (far right).",
    "input_schema": {
        "type": "object",
        "properties": {"angle": {"type": "integer", "description": "0 to 180"}},
        "required": ["angle"],
        "additionalProperties": False,
    },
    "strict": True,
}

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

# send the question to Claude, print the reply as it streams in, and return the finished message
def ask_claude(client: anthropic.Anthropic, question: Question) -> anthropic.types.Message:
    with client.messages.stream(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        output_config={"effort": "low"}, # fast replies; raise for harder questions
        system=SYSTEM_PROMPT,
        tools=[MOVE_SERVO_TOOL],
        messages=[{"role": "user", "content": build_content(question)}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
        final = stream.get_final_message()

    print()
    if final.stop_reason == "refusal":
        print("(Claude declined to answer)")
    return final

# carry out Claude's reply: move the servo if asked, then say the text out loud
def act_on_reply(reply: anthropic.types.Message, speaking: threading.Event) -> None:
    text = " ".join(b.text.strip() for b in reply.content if b.type == "text").strip()

    for block in reply.content:
        if block.type == "tool_use" and block.name == "move_servo":
            move_servo(block.input["angle"])
            text = text or SERVO_FALLBACK_SPEECH # Claude often calls the tool without saying anything

    if text:
        speaking.set() # the audio thread ignores the mic until we clear this
        try:
            speak(text)
            time.sleep(SPEAKER_ECHO_TAIL_SECONDS) # let the room echo die down
        finally:
            speaking.clear()

# LLM thread: question queue -> Claude -> printed reply, servo move and speech
def llm_loop(q: queue.Queue, client: anthropic.Anthropic, speaking: threading.Event, event: threading.Event) -> None:
    while not event.is_set():
        try:
            question = q.get(timeout=0.1)
        except queue.Empty:
            continue

        try:
            act_on_reply(ask_claude(client, question), speaking)
        except anthropic.APIError as e: # the SDK already retries transient errors
            print(f"\nClaude request failed: {e}")
