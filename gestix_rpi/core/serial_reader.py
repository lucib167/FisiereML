"""
Citire serial de la hub-ul ESP32-C3 (Gestix 2.0 RPi).

Format intrare (115200 baud, 100 Hz):
  Header (#...) — ignorat
  Date: TS,21×D,21×S (43 valori)  → extrage doar mâna DREAPTĂ (21)
  Status: # STATUS: D=ONLINE | S=ONLINE | ...

Emite prin pyqtSignal:
  frame_received(int, list)  — (timestamp_ms, 21 senzori mâna dreaptă)
  status_received(dict)
  connection_changed(bool)
"""

import re
import time
from collections import deque
from typing import List, Optional, Tuple

import serial
import serial.tools.list_ports
from PyQt5.QtCore import QThread, pyqtSignal

FRAME_SIZE_FULL = 43   # timestamp + 42 senzori (ambele mâini)
SENSORS_USED = 21      # doar mâna dreaptă (15 Hall + 6 IMU)
DEFAULT_BAUD = 115200
ONLINE_WINDOW_MS = 1500

_STATUS_RE = re.compile(r"^\s*#\s*STATUS:\s*(.+)$", re.IGNORECASE)


def parse_frame(line: str) -> Optional[Tuple[int, List[int]]]:
    """Returnează (timestamp, [21 senzori dreapta]) sau None."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split(",")
    if len(parts) != FRAME_SIZE_FULL:
        return None
    try:
        vals = [int(v) for v in parts]
    except ValueError:
        return None
    # vals[0] = timestamp, vals[1:22] = mâna dreaptă (21 senzori)
    return vals[0], vals[1:1 + SENSORS_USED]


def parse_status(line: str) -> Optional[dict]:
    m = _STATUS_RE.match(line.strip())
    if not m:
        return None
    out = {}
    for pair in m.group(1).split("|"):
        pair = pair.strip()
        if "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        v = v.strip().rstrip("Hz").strip()
        try:
            out[k.strip()] = float(v)
        except ValueError:
            out[k.strip()] = v
    return out


class SerialReader(QThread):
    frame_received     = pyqtSignal(int, list)
    status_received    = pyqtSignal(dict)
    connection_changed = pyqtSignal(bool)
    error_occurred     = pyqtSignal(str)

    def __init__(self, port: str = "", baud: int = DEFAULT_BAUD, parent=None):
        super().__init__(parent)
        self._port = port
        self._baud = baud
        self._running = False
        self._serial: Optional[serial.Serial] = None
        self._buffer: deque = deque(maxlen=500)
        self.right_online = False
        self._last_right_ms = 0.0

    def configure(self, port: str, baud: int = DEFAULT_BAUD):
        self._port = port
        self._baud = baud

    @staticmethod
    def list_ports() -> List[str]:
        return [p.device for p in serial.tools.list_ports.comports()]

    def get_latest_frame(self) -> Optional[Tuple[int, List[int]]]:
        try:
            return self._buffer[-1]
        except IndexError:
            return None

    def stop(self):
        self._running = False
        self.wait(3000)
        if self._serial and self._serial.is_open:
            try:
                self._serial.close()
            except Exception:
                pass

    def run(self):
        self._running = True
        if not self._port:
            self.error_occurred.emit("Niciun port serial configurat")
            return
        try:
            self._serial = serial.Serial(self._port, self._baud, timeout=1.0)
        except serial.SerialException as exc:
            self.error_occurred.emit(f"Nu pot deschide {self._port}: {exc}")
            return

        self.connection_changed.emit(True)
        try:
            while self._running:
                try:
                    raw = self._serial.readline()
                except serial.SerialException as exc:
                    if self._running:
                        self.error_occurred.emit(f"Eroare serial: {exc}")
                    break
                if not raw:
                    continue
                try:
                    line = raw.decode("ascii", errors="ignore")
                except Exception:
                    continue
                self._handle_line(line)
        finally:
            if self._serial and self._serial.is_open:
                self._serial.close()
        self.connection_changed.emit(False)

    def _handle_line(self, line: str):
        status = parse_status(line)
        if status is not None:
            self.status_received.emit(status)
            return
        parsed = parse_frame(line)
        if parsed is None:
            return
        ts_ms, sensors = parsed
        self._buffer.append((ts_ms, sensors))
        self.right_online = True
        self._last_right_ms = time.monotonic() * 1000
        self.frame_received.emit(ts_ms, sensors)
