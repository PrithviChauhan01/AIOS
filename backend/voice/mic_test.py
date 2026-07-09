"""Live mic input level meter — standalone diagnostic. No VAD, no STT.

Opens the same input device AIOS picks (voice.listener._pick_mic) at its native
sample rate and prints a live RMS + peak bar every ~100ms. Use it to confirm the
mic is alive and to eyeball speaking level before running the full voice loop.

Run from backend/:  python -m voice.mic_test
Ctrl+C to exit.
"""

import sys

import numpy as np
import sounddevice as sd

from voice.listener import _pick_mic, _device_samplerate

BAR_WIDTH = 18          # characters in the level bar
BAR_FULL_SCALE = 0.5    # amplitude that fills the bar (headroom above normal speech)
UPDATE_SECONDS = 0.1    # ~100ms per printed line


def _bar(value: float) -> str:
    filled = int(round(min(value / BAR_FULL_SCALE, 1.0) * BAR_WIDTH))
    return "#" * filled + "-" * (BAR_WIDTH - filled)


def main():
    dev = _pick_mic()
    device_sr = _device_samplerate(dev)
    try:
        name = sd.query_devices(dev if dev is not None else sd.default.device[0], "input")["name"]
    except Exception:
        name = "default"

    print(f"[mic_test] device='{name}' samplerate={device_sr}Hz — Ctrl+C to exit")

    frames = int(UPDATE_SECONDS * device_sr)
    try:
        with sd.InputStream(samplerate=device_sr, channels=1, dtype="int16",
                            blocksize=0, device=dev) as stream:
            while True:
                data, _ = stream.read(frames)
                x = data.flatten().astype(np.float32) / 32768.0
                if x.size == 0:
                    continue
                rms = float(np.sqrt(np.mean(x ** 2)))
                peak = float(np.abs(x).max())
                # \r keeps the meter on one line; overwrite the previous reading.
                sys.stdout.write(f"\rRMS {rms:.3f} |{_bar(rms)}| peak {peak:.2f}   ")
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n[mic_test] done.")
        sys.exit(0)


if __name__ == "__main__":
    main()
