#!/usr/bin/env python3
"""
GESTIX 2.0 — Interfața Raspberry Pi (entry point).

Pornește aplicația fullscreen, conectează serial-ul la hub și
afișează interfața de traducere limbaj semne ↔ voce.

Rulare:
    python3 main.py

Setări în config.py (port serial, căi modele, fullscreen).
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

import config as C
from core.serial_reader import SerialReader
from ui.interface import GestixWindow


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    app.setApplicationName("Gestix 2.0")

    # ── Serial ──
    port = C.detect_serial_port()
    reader = None
    if port:
        print(f"[Serial] Port detectat: {port}")
        reader = SerialReader(port, C.SERIAL_BAUD)
        reader.error_occurred.connect(lambda m: print(f"[Serial] {m}"))
        reader.start()
    else:
        print("[Serial] Niciun port găsit — UI pornește fără date live.")

    # ── Fereastra ──
    win = GestixWindow(serial_reader=reader)

    if C.FULLSCREEN:
        win.showFullScreen()
    else:
        win.resize(C.SCREEN_W, C.SCREEN_H)
        win.show()

    # Info backend STT
    print("[STT] Apasă pe toggle 'STT' pentru a porni microfonul.")
    print("[INFO] ESC = ieșire din fullscreen.")

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
