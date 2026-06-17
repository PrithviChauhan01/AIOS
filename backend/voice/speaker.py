from kokoro_onnx import Kokoro
import sounddevice as sd
import os

_kokoro = None

# Mood → speech delivery. Tune these as you like.
MOOD_SETTINGS = {
    "warm":    {"speed": 0.92},
    "neutral": {"speed": 0.90},
    "sharp":   {"speed": 0.98},
    "soft":    {"speed": 0.82},
}

def _get_kokoro():
    global _kokoro
    if _kokoro is None:
        model_path = os.path.join(os.path.dirname(__file__), "..", "kokoro-v1.0.onnx")
        voices_path = os.path.join(os.path.dirname(__file__), "..", "voices-v1.0.bin")
        _kokoro = Kokoro(model_path, voices_path)
    return _kokoro

def speak(text: str, mood: str = "neutral"):
    if not text:
        return
    settings = MOOD_SETTINGS.get(mood, MOOD_SETTINGS["neutral"])
    try:
        kokoro = _get_kokoro()
        samples, sample_rate = kokoro.create(
            text,
            voice="bf_emma",
            speed=settings["speed"],
            lang="en-gb"
        )
        sd.play(samples, sample_rate)
        sd.wait()
    except Exception as e:
        print(f"[speaker] TTS failed: {e}")
        print(f"AIOS: {text}")