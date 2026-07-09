"""Wake word + speech capture.

openWakeWord listens continuously for the wake word on the mic. On trigger, TEN VAD
captures speech from the same stream until a stretch of silence, then returns the buffer.

Both models load once, lazily, and run small/CPU-friendly so whisper keeps the VRAM."""
import queue
import time
from collections import deque

import numpy as np
import sounddevice as sd

# Diagnostics from the most recent _capture_speech run (additive — no effect on
# what gets captured). The voice loop reads this to log where time went and why a
# capture may not have registered. Single capture at a time, so a module dict is safe.
last_capture = {}

_last_logged_mic = None  # last (index, name, api) we logged — so we print the pick once
MIC_SILENCE_FLOOR = 0.001  # startup mean-amp below this = mic effectively dead → warn loudly


def _score_input(name: str, api: str) -> int:
    """Higher = better input. AirPods win outright; otherwise prefer WASAPI, then the
    ordinary user-space APIs. (WDM-KS is excluded before scoring — see _pick_mic.)"""
    score = 0
    if "airpods" in name:
        score += 1000
    if "wasapi" in api:
        score += 100
    elif "directsound" in api:
        score += 50
    elif "mme" in api:
        score += 25
    return score


def _pick_mic():
    """Pick the best available INPUT device. Prefer AirPods; if absent, don't fall back
    to a near-dead default — choose the strongest real input (WASAPI preferred, WDM-KS
    skipped, as it often opens but never delivers samples). Logs the pick once (and again
    only if the selection changes, e.g. AirPods connect later)."""
    global _last_logged_mic
    best = None  # (score, index, name, api)
    try:
        hostapis = sd.query_hostapis()
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] <= 0:
                continue
            ha = hostapis[d["hostapi"]]
            api = ha["name"].lower()
            if "wdm-ks" in api:
                continue  # kernel-streaming: opens but frequently yields silence
            score = _score_input(d["name"].lower(), api)
            # Among equal candidates, favour the one the OS marks as that API's default
            # input — i.e. "the default WASAPI input" when no AirPods are present.
            if ha.get("default_input_device") == i:
                score += 10
            if best is None or score > best[0]:
                best = (score, i, d["name"], api)
    except Exception as e:
        print(f"[mic] device enumeration failed: {e}")

    if best is None:
        if _last_logged_mic != ("default",):
            print("[mic] no usable input device found — falling back to system default")
            _last_logged_mic = ("default",)
        return None

    sig = (best[1], best[2], best[3])
    if sig != _last_logged_mic:
        print(f"[mic] selected device index={best[1]} name='{best[2]}' hostapi='{best[3]}'")
        _last_logged_mic = sig
    return best[1]


def _log_input_level(src):
    """Drain ~0.5s off the callback queue (via the 16k wrapper) and print mean amplitude.
    A dead/muted mic shows here immediately (~0.000) instead of looking like a VAD bug
    downstream. Reads through _Stream16k because the stream is now callback-driven — you
    can't blocking-read a callback stream."""
    try:
        data = src.read(int(0.5 * SR))  # 0.5s at 16k, served from the queue
        amp = float(np.abs(data.astype(np.float32) / 32768.0).mean())
        print(f"[mic] level={amp:.4f}")
        if amp < MIC_SILENCE_FLOOR:
            print("[mic] WARNING: input level very low — check mic/AirPods")
        return amp
    except Exception as e:
        print(f"[mic] level check failed: {e}")
        return None


sd.default.device = (_pick_mic(), None)

SR = 16000
OWW_FRAME = 1280          # 80ms @ 16k — openWakeWord's expected frame size
HOP_SIZE = 256            # TEN VAD requires exactly 256 samples (16ms @ 16k) per frame
WAKE_WORD = "alexa"  # bundled openWakeWord model; swap for another pretrained name
WAKE_THRESHOLD = 0.4

# TEN VAD endpointing. Threshold is the per-frame speech-probability gate; frames below
# it count toward trailing silence. Min-silence closes the turn ~1100ms after you stop
# talking — long enough that a brief natural pause mid-sentence doesn't clip the end of
# an utterance; only sustained silence ends the capture. Max-capture is a *wall-clock*
# cap so a missed speech-end can't block the loop the way a chunk-count cap did under
# slow CPU inference.
VAD_THRESHOLD = 0.5           # speech-probability gate for onset / continuation (tunable)
VAD_MIN_SILENCE_MS = 1100     # trailing silence that ends a capture (was 600 — clipped pauses)
MAX_CAPTURE_SECONDS = 15      # wall-clock cap on a single utterance

# Captured speech comes in very quiet → Whisper hallucinates on near-silence, so we
# peak-normalize each utterance to a target (audible, never clipped). TEN VAD decides
# speech vs silence per frame, so a captured (non-empty) buffer is trusted as real speech
# and only an empty buffer (VAD never fired) is dropped — no absolute amplitude floor,
# which was miscalibrated for this quiet mic and wrongly zeroed real captures to empty.
NORM_TARGET = 0.7        # peak-normalization target amplitude
PREROLL_SECONDS = 0.4    # audio kept from before VAD fires, so opening words survive
# Real-speech gate measured on the RAW capture, BEFORE normalization. Normalization
# scales even quiet room-tone up to NORM_TARGET, so the silence decision MUST happen on
# the un-normalized signal — otherwise every near-silent buffer passes and Whisper
# transcribes amplified noise as "Yeah". Tune so normal speech passes, breath/room-tone
# is rejected; watch the logged raw_rms to calibrate.
SPEECH_RMS_FLOOR = 0.01  # raw RMS below this = no speech, skip

_oww = None
_vad = None


def _get_oww():
    global _oww
    if _oww is None:
        import openwakeword
        from openwakeword.model import Model
        try:
            openwakeword.utils.download_models()  # no-op if already present
        except Exception as e:
            print(f"[listener] oww model download skipped ({e})")
        # onnx runtime is the portable choice on Windows (tflite-runtime isn't on pip).
        _oww = Model(wakeword_models=[WAKE_WORD], inference_framework="onnx")
    return _oww


def _get_vad():
    global _vad
    if _vad is None:
        from ten_vad import TenVad
        # One instance reused across captures (TEN VAD has no reset API; its per-frame
        # state settles during the silence between turns). DLL load is cached by the OS.
        _vad = TenVad(hop_size=HOP_SIZE, threshold=VAD_THRESHOLD)
        print(
            f"[listener] TEN VAD: threshold={VAD_THRESHOLD} hop={HOP_SIZE} "
            f"min_silence={VAD_MIN_SILENCE_MS}ms"
        )
    return _vad


def _reset_oww(oww):
    try:
        oww.reset()
    except Exception:
        try:
            oww.prediction_buffer.clear()
        except Exception:
            pass


_scipy_resample = None
_scipy_checked = False


def _device_samplerate(device) -> int:
    """The rate to OPEN the stream at — the device's native default. Many phone/USB mics
    are 44100-only and throw PortAudioError -9997 if you force 16000. Falls back to 44100,
    then 16000, if the query fails."""
    try:
        idx = device if device is not None else sd.default.device[0]
        info = sd.query_devices(idx, "input")
        sr = int(round(info.get("default_samplerate", 0)))
        if sr > 0:
            return sr
    except Exception as e:
        print(f"[mic] samplerate query failed ({e}); assuming 44100")
    return 44100


def _resample_to_16k(block: np.ndarray, device_sr: int) -> np.ndarray:
    """Downsample an int16 mono block from device_sr to SR (16k). Uses scipy's polyphase
    resampler (resample_poly) when available, else a numpy linear-interp fallback so it
    works even where scipy has no wheel (e.g. Python 3.14). Returns int16."""
    global _scipy_resample, _scipy_checked
    if device_sr == SR or block.size == 0:
        return block.astype(np.int16)
    if not _scipy_checked:
        _scipy_checked = True
        try:
            from scipy.signal import resample_poly
            _scipy_resample = resample_poly
        except Exception:
            _scipy_resample = None
    x = block.astype(np.float32)
    if _scipy_resample is not None:
        from math import gcd
        g = gcd(int(device_sr), SR)
        out = _scipy_resample(x, SR // g, int(device_sr) // g)
    else:
        n_out = int(round(x.size * SR / device_sr))
        out = np.interp(np.linspace(0, x.size - 1, max(n_out, 1)), np.arange(x.size), x)
    return np.clip(np.round(out), -32768, 32767).astype(np.int16)


def _open_callback_stream(dev, device_sr):
    """Open a callback-mode InputStream and return (stream, queue). Each PortAudio block
    is copied into the queue from the audio thread; _Stream16k drains it. Callback delivery
    decouples us from PortAudio's blocking read(), which stalled mid-capture on Bluetooth
    (WASAPI AirPods) — the read would hang, the meter froze, and frames were dropped so VAD
    missed onset. A steady 30ms native block + high latency absorbs the Bluetooth jitter."""
    q = queue.Queue()

    def _callback(indata, frames, time_info, status):
        if status:
            print(f"[mic] stream status: {status}")
        # Copy — PortAudio reuses indata's buffer after the callback returns.
        q.put(indata[:, 0].copy())

    stream = sd.InputStream(samplerate=device_sr, channels=1, dtype="int16",
                            blocksize=int(device_sr * 0.03), latency="high",
                            device=dev, callback=_callback)
    return stream, q


class _Stream16k:
    """Drains a callback queue of native-rate int16 blocks and serves fixed-size int16
    frames AT 16k, resampling on the fly. Both TEN VAD (256) and openWakeWord (1280) require
    16k frames, so this keeps that contract no matter the device's native rate."""

    _UNDERFLOW_TIMEOUT = 0.5  # seconds to wait for a block before inserting silence

    def __init__(self, q, device_sr):
        self.q = q
        self.device_sr = int(round(device_sr))
        self._silence_block = np.zeros(int(self.device_sr * 0.03), dtype=np.int16)
        self._pending = np.zeros(0, dtype=np.int16)

    def read(self, n: int) -> np.ndarray:
        """Return exactly n int16 samples at 16k, pulling/resampling queued native blocks.
        On queue underflow (a Bluetooth stall) insert one native block of silence rather
        than hang — VAD reads it as silence and real speech resumes when blocks flow again."""
        while self._pending.size < n:
            try:
                block = self.q.get(timeout=self._UNDERFLOW_TIMEOUT).astype(np.int16)
            except queue.Empty:
                block = self._silence_block
            if self.device_sr != SR:
                block = _resample_to_16k(block, self.device_sr)
            self._pending = np.concatenate([self._pending, block])
        frame, self._pending = self._pending[:n], self._pending[n:]
        return frame


def _capture_speech(src) -> np.ndarray:
    """Read 256-sample frames through TEN VAD until trailing silence. float32 [-1,1].

    src is a _Stream16k draining the callback queue: the stream is opened at the device's
    NATIVE rate (e.g. 44100) and _Stream16k resamples each queued block down to SR=16000,
    serving exactly HOP_SIZE (256) int16 samples per frame — the shape/rate TEN VAD requires.
    process() returns (probability, flag). Callers share one src so listen()'s wake-word
    buffer isn't dropped between wake detection and capture."""
    vad = _get_vad()

    collected = []
    # Rolling buffer of pre-VAD frames. VAD fires a beat after speech onset, so the
    # opening word(s) land here; we prepend it once speech starts so they aren't lost.
    preroll = deque(maxlen=int(PREROLL_SECONDS * SR / HOP_SIZE))
    speech_started = False
    silence_frames = 0
    silence_limit = int((VAD_MIN_SILENCE_MS / 1000.0) * SR / HOP_SIZE)
    # Generous frame ceiling; the wall-clock check below is the real cap.
    max_frames = int((MAX_CAPTURE_SECONDS + 5) * SR / HOP_SIZE)

    # Energy fallback: TEN VAD sometimes never fires on a quiet-but-present voice
    # (notably when AirPods hand off / come back weak). So a frame counts as speech if
    # VAD fires OR its raw energy clears the speech floor — measured on the int16 scale
    # to match SPEECH_RMS_FLOOR (which is on the float32 [-1,1] scale). Without this a
    # VAD miss yields an empty buffer that looks like a VAD bug.
    energy_floor_int16 = SPEECH_RMS_FLOOR * 32768.0
    energy_fallback = False  # diagnostic: did energy carry the onset where VAD didn't?

    t_start = time.perf_counter()
    t_speech = None          # when speech onset was detected (VAD or energy)
    ended_on_silence = False  # broke out on trailing silence vs hit the cap

    for _ in range(max_frames):
        frame = src.read(HOP_SIZE)  # int16, 256 samples @16k — resampled from native by _Stream16k

        prob, _flag = vad.process(frame)
        frame_rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))

        vad_active = prob >= VAD_THRESHOLD
        energy_active = frame_rms >= energy_floor_int16

        if vad_active or energy_active:
            if not speech_started:
                t_speech = time.perf_counter()
                collected.extend(preroll)  # prepend pre-roll so opening words survive
                preroll.clear()
            if energy_active and not vad_active:
                energy_fallback = True  # VAD missed this frame; energy kept us capturing
            speech_started = True
            silence_frames = 0
            collected.append(frame)
        elif speech_started:
            collected.append(frame)
            silence_frames += 1
            if silence_frames >= silence_limit:
                ended_on_silence = True
                break
        else:
            preroll.append(frame)  # not speaking yet — keep only the rolling pre-roll

        # Wall-clock cap: bound on elapsed real time so a missed speech-end can't block.
        if time.perf_counter() - t_start >= MAX_CAPTURE_SECONDS:
            break

    t_end = time.perf_counter()
    # Concatenate pre-roll + speech frames into the utterance. Frames are only appended
    # to `collected` at/after speech onset, so a non-empty list means speech was detected
    # (by TEN VAD or the energy fallback); an empty list means neither ever fired.
    audio = (
        np.concatenate(collected).astype(np.float32) / 32768.0
        if collected else np.zeros(0, dtype=np.float32)
    )

    # Real-speech gate on the RAW signal, measured BEFORE normalization. (Normalizing
    # first would scale room-tone up to NORM_TARGET and let every near-silent buffer
    # through — that is what made Whisper hallucinate "Yeah" on near-silence.)
    raw_rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    if audio.size == 0:
        # VAD never detected speech — nothing to transcribe.
        print("[skip] no speech detected")
    elif raw_rms < SPEECH_RMS_FLOOR:
        # VAD fired but the raw energy is room-tone/breath, not speech.
        print(f"[skip] no-speech raw_rms={raw_rms:.4f}")
        audio = np.zeros(0, dtype=np.float32)
    else:
        # Peak-normalize the captured speech so the loudest sample hits NORM_TARGET —
        # audible, never clipped. Only buffers that passed the RAW gate get here.
        peak = float(np.abs(audio).max())
        if peak > 0:
            audio = audio * (NORM_TARGET / peak)

    # Diagnostics only — split the wait-for-speech phase from the active recording,
    # and record whether VAD ended cleanly on silence or ran into the hard cap.
    if t_speech is not None:
        vad_wait_ms = (t_speech - t_start) * 1000.0
        capture_ms = (t_end - t_speech) * 1000.0
    else:
        vad_wait_ms = (t_end - t_start) * 1000.0  # never fired — all of it was waiting
        capture_ms = 0.0
    last_capture.clear()
    last_capture.update(
        vad_fired=speech_started,
        energy_fallback=energy_fallback,  # True → energy floor caught speech VAD missed
        timed_out=not ended_on_silence,
        vad_wait_ms=vad_wait_ms,
        capture_ms=capture_ms,
        raw_rms=raw_rms,  # pre-normalization energy — transcriber uses it to drop filler
    )

    # Confirm the buffer handed to the transcriber is non-empty and carries signal,
    # and show the raw RMS the gate decided on so the floor can be calibrated.
    final_amp = float(np.abs(audio).max()) if audio.size else 0.0
    print(f"[capture] final buffer: len={audio.size} samples "
          f"({audio.size / SR:.2f}s) raw_rms={raw_rms:.4f} max_amp={final_amp:.3f}"
          f"{' [energy-fallback]' if energy_fallback else ''}")
    return audio


def capture() -> np.ndarray:
    """Capture one utterance via VAD only — no wake word. Opens its own mic stream at the
    device's native rate, records until trailing silence, returns float32 16k mono
    (resampled). Used by toggle mode."""
    dev = _pick_mic()
    device_sr = _device_samplerate(dev)
    print(f"[mic] device_sr={device_sr} -> resampling to {SR}")
    stream, q = _open_callback_stream(dev, device_sr)
    with stream:
        src = _Stream16k(q, device_sr)
        _log_input_level(src)
        return _capture_speech(src)


def listen() -> np.ndarray:
    """Block until the wake word fires, then capture the command. Returns float32 16k mono."""
    oww = _get_oww()
    _reset_oww(oww)

    dev = _pick_mic()
    device_sr = _device_samplerate(dev)
    print(f"[mic] device_sr={device_sr} -> resampling to {SR}")
    stream, q = _open_callback_stream(dev, device_sr)
    with stream:
        src = _Stream16k(q, device_sr)  # serves 16k frames to oww too — it needs 16k like TEN VAD
        _log_input_level(src)
        while True:
            frame = src.read(OWW_FRAME)  # int16, 1280 samples @16k
            scores = oww.predict(frame)
            if scores.get(WAKE_WORD, 0.0) >= WAKE_THRESHOLD:
                break
        print("[listener] wake word detected — listening...")
        # 2. capture speech until silence, reusing the open stream AND its resampler buffer
        return _capture_speech(src)
