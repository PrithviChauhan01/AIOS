"""Speaker verification via Resemblyzer. Loads the enrolled voiceprint and answers
one question: is this Prithvi? Cosine similarity against the stored embedding.

Runs on CPU on purpose — Resemblyzer is light, and the VRAM is reserved for whisper/TTS."""

import os
import numpy as np

THRESHOLD = 0.60  # tune: higher = stricter
_VP_PATH = os.path.join(os.path.dirname(__file__), "voiceprint.npy")

_encoder = None
_voiceprint = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        from resemblyzer import VoiceEncoder
        _encoder = VoiceEncoder("cpu")
    return _encoder


def _get_voiceprint():
    global _voiceprint
    if _voiceprint is None:
        if not os.path.exists(_VP_PATH):
            raise FileNotFoundError(
                f"no voiceprint at {_VP_PATH} — run `python voice/enroll.py` first"
            )
        _voiceprint = np.load(_VP_PATH)
    return _voiceprint


def embed(audio, sr: int = 16000):
    """audio: float32 mono numpy. Returns an L2-normalized speaker embedding."""
    from resemblyzer import preprocess_wav
    wav = preprocess_wav(audio, source_sr=sr)
    return _get_encoder().embed_utterance(wav)


def verify(audio, sr: int = 16000, threshold: float = THRESHOLD) -> bool:
    """True iff the audio matches the enrolled voiceprint above threshold."""
    try:
        emb = embed(audio, sr)
        ref = _get_voiceprint()
        sim = float(np.dot(emb, ref) / (np.linalg.norm(emb) * np.linalg.norm(ref)))
        print(f"[verify] similarity={sim:.3f} (threshold={threshold})")
        return sim >= threshold
    except Exception as e:
        print(f"[verify] failed ({e}) → rejecting")
        return False
