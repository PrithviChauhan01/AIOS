"""One-time enrollment. Records ~5 samples of Prithvi speaking via the mic, builds a
Resemblyzer voiceprint (mean of per-sample embeddings) and saves it to voice/voiceprint.npy.

Run standalone:  python voice/enroll.py
"""

import os
import numpy as np
import sounddevice as sd


def _pick_mic():
    try:
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0 and "airpods" in d["name"].lower():
                return i
    except Exception:
        pass
    return None


sd.default.device = (_pick_mic(), None)

SR = 16000
DURATION = 4          # seconds per sample
NUM_SAMPLES = 5
_VP_PATH = os.path.join(os.path.dirname(__file__), "voiceprint.npy")


def _record(seconds: int) -> np.ndarray:
    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def main():
    from resemblyzer import VoiceEncoder, preprocess_wav
    encoder = VoiceEncoder("cpu")

    print(f"Enrollment — speak naturally for ~{DURATION}s, {NUM_SAMPLES} times.")
    print("Say something different each round (count, read a sentence, etc).\n")

    embeds = []
    for i in range(NUM_SAMPLES):
        input(f"[{i + 1}/{NUM_SAMPLES}] Press Enter, then speak...")
        audio = _record(DURATION)
        wav = preprocess_wav(audio, source_sr=SR)
        embeds.append(encoder.embed_utterance(wav))
        print("  captured.\n")

    voiceprint = np.mean(embeds, axis=0)
    voiceprint = voiceprint / np.linalg.norm(voiceprint)
    np.save(_VP_PATH, voiceprint)
    print(f"Done. Voiceprint saved → {_VP_PATH}")


if __name__ == "__main__":
    main()
