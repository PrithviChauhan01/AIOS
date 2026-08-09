"""STT — Groq hosted Whisper (primary) with local faster-whisper as fallback.

Primary: Groq's OpenAI-compatible audio.transcriptions.create on
'whisper-large-v3-turbo' — fast, accurate, uses the existing GROQ_API_KEY.
Fallback: the local 'distil-small.en' CTranslate2 model on CPU (int8), loaded
once and lazily, kept lean for the 4GB VRAM budget shared with the other models.
The local repo is the Systran CT2 conversion; faster-whisper can't load the
plain Transformers distil-whisper repo.

Only the transcription step lives here — capture, TEN VAD, normalization and the
silence/filler gate are untouched (listener.py / loop.py own those).
"""

import os
import wave
import tempfile

import numpy as np

_model = None
_groq = None

# Phrases Whisper hallucinates on low-energy audio. If the capture was quiet (raw RMS
# below this ceiling) and the whole transcript is nothing but one of these, it's noise,
# not speech — drop it. The ceiling sits just above the listener's hard RMS floor so it
# only second-guesses the borderline band, never a clearly-voiced utterance.
_FILLER = {"yeah", "thank you", "thanks", "you", "okay", "ok", "bye", ""}
FILLER_RMS_CEILING = 0.02

# Groq hosted-Whisper config. The prompt biases decoding toward the names AIOS
# would otherwise mis-hear ("AIOS", "Sir").
_GROQ_MODEL = "whisper-large-v3-turbo"
_GROQ_PROMPT = (
    "Conversation with AIOS, an AI assistant. The user is called Sir. "
    "Names: AIOS, Sir."
)


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel("Systran/faster-distil-whisper-small.en", device="cpu", compute_type="int8")
        print("[transcriber] faster-whisper 'Systran/faster-distil-whisper-small.en' on cpu (int8)")
    return _model


def _get_groq():
    global _groq
    if _groq is None:
        from groq import Groq
        from config import Config
        from core.net import LONG_TIMEOUT
        # Long read budget: this uploads a wav and waits for Whisper, not a token
        # stream. Bounded all the same — the local faster-whisper fallback below is
        # only useful if the hosted call actually gives up.
        _groq = Groq(api_key=Config.GROQ_API_KEY, timeout=LONG_TIMEOUT, max_retries=0)
    return _groq


def _write_wav(audio) -> str:
    """Dump the float32 [-1,1] mono buffer to a temp 16kHz mono 16-bit PCM wav and
    return its path. Groq wants a file, not a numpy array; the local path already
    has the array, so this is only on the Groq branch. Caller removes the file."""
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="aios_stt_")
    os.close(fd)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)        # 16-bit
        wf.setframerate(16000)    # 16kHz, matching the capture rate
        wf.writeframes(pcm16.tobytes())
    return path


def _transcribe_groq(audio) -> str:
    """Primary path. Sends the captured audio to Groq hosted Whisper. Raises on any
    failure (network / rate-limit / SDK error) so transcribe() can fall back."""
    client = _get_groq()
    path = _write_wav(audio)
    try:
        with open(path, "rb") as f:
            result = client.audio.transcriptions.create(
                file=f,
                model=_GROQ_MODEL,
                language="en",
                temperature=0.0,
                response_format="text",
                prompt=_GROQ_PROMPT,
            )
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    # response_format="text" returns the raw string; guard for SDK variants that
    # still wrap it in an object with a .text attribute.
    text = result if isinstance(result, str) else getattr(result, "text", str(result))
    return text.strip()


def _transcribe_local(audio) -> str:
    """Fallback path. Local faster-whisper, identical decode settings to before."""
    model = _get_model()
    segments, _ = model.transcribe(
        audio,
        language="en",
        beam_size=1,
        initial_prompt="This is a conversation with AIOS. The user calls her AIOS and she calls him Sir. AIOS, AIOS.",
        no_speech_threshold=0.6,            # suppress hallucinated phrases on silence
        condition_on_previous_text=False,   # don't let prior text seed phantom output
    )
    return " ".join(s.text.strip() for s in segments).strip()


def transcribe(audio, raw_rms=None, force_local=False) -> str:
    """audio: float32 mono numpy at 16kHz, normalized to [-1, 1]. Returns text.
    raw_rms: the capture's pre-normalization RMS (from the listener), used to drop
    filler-token hallucinations on low-energy audio. None disables that guard.
    force_local: secret-tier guard — when True the audio NEVER leaves the machine;
    skip Groq entirely and run the local model only."""
    if audio is None or audio.size == 0:
        return ""  # listener dropped it as silence — nothing to transcribe

    text = ""
    engine = None
    if not force_local:
        try:
            text = _transcribe_groq(audio)
            engine = "groq"
        except Exception as e:
            print(f"[transcriber] groq failed ({type(e).__name__}: {e}) -- falling back to local")

    if engine is None:
        text = _transcribe_local(audio)
        engine = "local"

    print(f"[transcriber] engine={engine}")

    # Drop pure-filler hallucinations when the capture was quiet. Whisper still emits
    # "Yeah"/"Thank you"/"you" on low-energy audio that slips past no_speech_threshold;
    # if raw energy was low AND the entire transcript is just filler, treat it as noise.
    if raw_rms is not None and raw_rms < FILLER_RMS_CEILING:
        if text.lower().strip(" .!?,") in _FILLER:
            return ""
    return text
