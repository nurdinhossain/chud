import requests

from config import SERVO_URL, SERVO_MIN_ANGLE, SERVO_MAX_ANGLE, ESP_TIMEOUT_SECONDS

# ask the servo ESP32 to move to an angle
def move_servo(angle: int) -> None:
    angle = max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, angle))
    try:
        res = requests.post(SERVO_URL, params={"angle": angle}, timeout=ESP_TIMEOUT_SECONDS)
        res.raise_for_status()
    except requests.RequestException as e:
        print(f"Servo request failed: {e}")
