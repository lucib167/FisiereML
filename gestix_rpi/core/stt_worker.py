"""
STT Worker — voce → text offline (Whisper), pentru RPi.

Preferă faster-whisper (eficient pe ARM). Fallback la openai-whisper.
Dacă niciunul nu e instalat → mod simulare (fraze demo).

Semnale:
  status(str)   — stare curentă
  final(str)    — text transcris
  level(float)  — nivel audio 0-1 (waveform)
"""

import math
import random
import threading
import time

from PyQt5.QtCore import QObject, pyqtSignal

# Detectare backend STT
_BACKEND = None
try:
    from faster_whisper import WhisperModel  # noqa: F401
    _BACKEND = "faster"
except ImportError:
    try:
        import whisper  # noqa: F401
        _BACKEND = "openai"
    except ImportError:
        _BACKEND = None

try:
    import sounddevice as sd
    import numpy as np
    _AUDIO_OK = True
except ImportError:
    _AUDIO_OK = False


class STTWorker(QObject):
    status = pyqtSignal(str)
    final  = pyqtSignal(str)
    level  = pyqtSignal(float)

    SAMPLE_RATE = 16000

    def __init__(self, model_name: str = "small", lang: str = "ro",
                 chunk_seconds: int = 5):
        super().__init__()
        self._model_name = model_name
        self._lang = lang
        self._chunk = chunk_seconds
        self._running = False
        self._model = None

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._running = False
        if _AUDIO_OK:
            try:
                sd.stop()
            except Exception:
                pass

    # ── Loop principal ───────────────────────────────────────

    def _loop(self):
        if _BACKEND is None or not _AUDIO_OK:
            self._simulate()
            return

        self.status.emit(f"Încarc model Whisper ({_BACKEND})...")
        try:
            if _BACKEND == "faster":
                from faster_whisper import WhisperModel
                self._model = WhisperModel(self._model_name, device="cpu",
                                            compute_type="int8")
            else:
                import whisper
                self._model = whisper.load_model(self._model_name)
        except Exception as exc:
            print(f"[STT] Eroare model: {exc}")
            self._simulate()
            return

        self.status.emit("Ascultă...")
        while self._running:
            try:
                audio = self._record_chunk()
                if not self._running:
                    break
                self.status.emit("Procesează...")
                text = self._transcribe(audio)
                if text and text not in {".", ",", "...", "Mulțumesc.", "Mulțumesc"}:
                    self.final.emit(text)
                self.status.emit("Ascultă...")
            except Exception as exc:
                print(f"[STT] {exc}")
                self.status.emit("Eroare microfon")
                break

    def _record_chunk(self):
        frames = int(self.SAMPLE_RATE * self._chunk)
        audio = sd.rec(frames, samplerate=self.SAMPLE_RATE,
                       channels=1, dtype="float32")
        step_ms = 100
        for _ in range(int(self._chunk * 1000 / step_ms)):
            if not self._running:
                sd.stop()
                break
            sd.sleep(step_ms)
            valid = audio[~np.isnan(audio[:, 0]), 0]
            lvl = float(np.abs(valid).mean() * 8) if len(valid) else 0.0
            self.level.emit(min(lvl, 1.0))
        sd.wait()
        return audio[:, 0]

    def _transcribe(self, audio) -> str:
        if _BACKEND == "faster":
            segments, _ = self._model.transcribe(
                audio, language=self._lang, beam_size=1)
            return " ".join(s.text for s in segments).strip()
        else:
            result = self._model.transcribe(
                audio, language=self._lang, fp16=False,
                condition_on_previous_text=False)
            return result.get("text", "").strip()

    def _simulate(self):
        self.status.emit("Ascultă... (simulare — Whisper neinstalat)")
        phrases = [
            "bună ziua",
            "cu ce vă pot ajuta",
            "mulțumesc foarte mult",
            "aveți o programare",
            "la revedere",
        ]
        t = 0.0
        next_at = random.uniform(4, 7)
        while self._running:
            t += 0.05
            self.level.emit(min(abs(math.sin(t * 1.4)) * 0.5 + random.random() * 0.1, 1.0))
            time.sleep(0.05)
            if t >= next_at:
                self.final.emit(random.choice(phrases))
                next_at = t + random.uniform(4, 7)
