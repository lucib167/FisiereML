"""
Configurare Gestix 2.0 RPi.

Editează aici căile și parametrii. Toate setările într-un singur loc.
"""

import os

# ── Serial (conexiune cu hub-ul ESP32-C3) ────────────────────
# Pe RPi/Linux: de obicei "/dev/ttyUSB0" sau "/dev/ttyACM0"
# Pe Windows (dezvoltare): "COM9"
# None = auto-detectare (caută primul port disponibil)
SERIAL_PORT = None
SERIAL_BAUD = 115200

# ── Model ML (recunoaștere semne) ────────────────────────────
# Lucian: pune aici căile către modelele tale antrenate.
# Lasă None pentru a porni fără model (UI funcționează, fără inferență).
MODEL_STATIC_PATH = None   # ex: "/home/pi/gestix/models/gestix_static.tflite"
MODEL_MOTION_PATH = None   # ex: "/home/pi/gestix/models/gestix_motion.tflite"
# .labels.json și _preprocessor.json se caută automat lângă model.

# ── Inferență ────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.80   # sub asta nu emite predicția
STRIDE_FRAMES        = 3      # rulează inferență la fiecare N frame-uri (~33 Hz)
GYRO_MOTION_THRESHOLD = 800   # gyro magnitude: peste = MOTION, sub = STATIC

# ── Sentence Builder (conversație) ───────────────────────────
WORD_GAP_SEC     = 2.0   # IDLE > 2s = separator între cuvinte
SENTENCE_END_SEC = 5.0   # IDLE > 5s = trimite propoziția în chat

# ── STT (Whisper) ────────────────────────────────────────────
WHISPER_MODEL = "small"   # "tiny" (rapid) | "small" (recomandat) | "base"
WHISPER_LANG  = "ro"
CHUNK_SECONDS = 5

# ── Display ──────────────────────────────────────────────────
FULLSCREEN = True         # True pe RPi (ecran 7"), False la dezvoltare pe PC
SCREEN_W = 800
SCREEN_H = 480

# ── MOD DEMO (prezentare scriptată) ──────────────────────────
# Când True, modul SEMNE NU folosește ML real — rulează un scenariu scriptat.
# Fiecare frază se declanșează când INTRI în modul SEMNE, după delay-ul ei.
# (prima frază pornește la start; următoarele la fiecare comutare STT→SEMNE)
DEMO_MODE = True
DEMO_SCRIPT = [
    ("salut",                  6.0),
    ("cum te cheamă",          6.0),
    ("numele meu este luci",  10.0),
]
DEMO_WORD_INTERVAL_MS = 600   # cât durează „semnarea" fiecărui cuvânt

# ── Paletă culori (identitate Gestix) ────────────────────────
BG    = "#131313"
CARD  = "#1c1c1c"
CARD2 = "#252525"
CARD3 = "#2e2e2e"
WHITE = "#f0f0f0"
GRAY  = "#6e6e6e"
GRAY2 = "#383838"

MODE_SIGN = "sign"
MODE_STT  = "stt"
ACCENT    = {MODE_SIGN: "#ff1461", MODE_STT: "#1461ff"}
ACCENT_DK = {MODE_SIGN: "#7a0030", MODE_STT: "#003080"}


def detect_serial_port() -> str:
    """Auto-detectează portul serial (RPi sau Windows)."""
    if SERIAL_PORT:
        return SERIAL_PORT
    try:
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        # Preferă ttyUSB/ttyACM pe Linux
        for p in ports:
            if "USB" in p.device or "ACM" in p.device:
                return p.device
        if ports:
            return ports[0].device
    except Exception:
        pass
    return ""
