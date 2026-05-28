"""
Interfața principală Gestix 2.0 — Raspberry Pi.

QMainWindow fullscreen 800×480 cu:
  - Toggle SEMNE / STT
  - Panel stâng: semn detectat (literă + confidence) sau microfon + waveform
  - Panel drept: conversație cu bule (SEMNE → dreapta roz, STT → stânga albastru)
  - Sentence builder: acumulează semne → propoziție → bulă

Integrează:
  - SerialReader (date de la hub)
  - SignInference (model ML — dual static/motion, comutare pe gyro)
  - SentenceBuilder (conversație)
  - STTWorker (Whisper)
"""

import math
import random
from collections import deque
from datetime import datetime
from typing import List, Optional

import numpy as np
from PyQt5.QtCore import QRect, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath
from PyQt5.QtWidgets import (
    QHBoxLayout, QLabel, QMainWindow, QPushButton, QScrollArea,
    QStackedWidget, QVBoxLayout, QWidget,
)

import config as C
from core.sentence_builder import SentenceBuilder, is_idle, format_class_name
from core.sign_inference import SignInference
from core.stt_worker import STTWorker

# Indici IMU pentru gyro magnitude (în frame de 21)
GYRO_IDX = [18, 19, 20]
EMA_ALPHA = 0.4
STABILITY_FRAMES = 3


# ── Toggle SEMNE/STT ─────────────────────────────────────────

class ModeToggle(QWidget):
    mode_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode = C.MODE_SIGN
        self.setFixedSize(150, 27)
        self.setCursor(Qt.PointingHandCursor)

    @property
    def mode(self):
        return self._mode

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        hw, r = w // 2, h / 2
        bg = QPainterPath()
        bg.addRoundedRect(0, 0, w, h, r, r)
        p.fillPath(bg, QColor(C.CARD2))
        p.setPen(QColor(C.GRAY2))
        p.drawPath(bg)
        p.setClipPath(bg)
        act = QPainterPath()
        act.addRect(0 if self._mode == C.MODE_SIGN else hw, 0, hw, h)
        p.fillPath(act, QColor(C.ACCENT[self._mode]))
        p.setClipping(False)
        p.setPen(QColor(C.GRAY2))
        p.drawLine(hw, 3, hw, h - 3)
        p.setFont(QFont("Segoe UI", 8, QFont.Bold))
        p.setPen(QColor(C.WHITE) if self._mode == C.MODE_SIGN else QColor(C.GRAY))
        p.drawText(QRect(0, 0, hw, h), Qt.AlignCenter, "SEMNE")
        p.setPen(QColor(C.WHITE) if self._mode == C.MODE_STT else QColor(C.GRAY))
        p.drawText(QRect(hw, 0, hw, h), Qt.AlignCenter, "STT")

    def mousePressEvent(self, evt):
        new = C.MODE_STT if evt.x() > self.width() // 2 else C.MODE_SIGN
        if new != self._mode:
            self._mode = new
            self.update()
            self.mode_changed.emit(self._mode)


# ── Waveform ─────────────────────────────────────────────────

class WaveformBars(QWidget):
    def __init__(self, bars=14, parent=None):
        super().__init__(parent)
        self._bars = bars
        self._levels = [0.04] * bars
        self.setMinimumHeight(48)

    def set_level(self, lvl):
        for i in range(self._bars):
            dist = abs(i - self._bars / 2) / (self._bars / 2)
            target = max(0.04, lvl * (1 - dist * 0.35) + random.uniform(-0.04, 0.04))
            self._levels[i] = self._levels[i] * 0.45 + target * 0.55
        self.update()

    def idle(self):
        for i in range(self._bars):
            self._levels[i] = max(0.04, self._levels[i] * 0.82)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        gap = 3
        bw = max(4, (w - self._bars * gap) // self._bars)
        col = QColor(C.ACCENT[C.MODE_STT])
        for i, lvl in enumerate(self._levels):
            bh = max(4, int(h * lvl))
            x = i * (bw + gap)
            y = (h - bh) // 2
            path = QPainterPath()
            path.addRoundedRect(x, y, bw, bh, bw // 2, bw // 2)
            p.fillPath(path, col)


# ── Fereastra principală ─────────────────────────────────────

class GestixWindow(QMainWindow):
    def __init__(self, serial_reader=None):
        super().__init__()
        self.setWindowTitle("GESTIX 2.0")
        self.setMinimumSize(C.SCREEN_W, C.SCREEN_H)
        self.setStyleSheet(self._stylesheet())

        self._reader = serial_reader
        self._mode = C.MODE_SIGN
        self._stt: Optional[STTWorker] = None

        # ── Inferență ──
        self._inf_static = SignInference()
        self._inf_motion = SignInference()
        self._active_mode = "static"
        self._buf_static: deque = deque(maxlen=1)
        self._buf_motion: deque = deque(maxlen=200)
        self._stride = 0
        self._probs_ema = None
        self._recent = deque(maxlen=6)

        # ── Sentence builder ──
        self._sb = SentenceBuilder(
            word_gap_sec=C.WORD_GAP_SEC,
            sentence_end_sec=C.SENTENCE_END_SEC,
            on_update=self._on_sentence_preview,
            on_sentence=self._on_sentence_complete,
        )

        self._build_ui()
        self._load_models()

        # Timers
        self._clock_tmr = QTimer(self); self._clock_tmr.timeout.connect(self._tick); self._clock_tmr.start(1000); self._tick()
        self._wave_tmr = QTimer(self); self._wave_tmr.timeout.connect(lambda: self._wave.idle())
        self._sb_tmr = QTimer(self); self._sb_tmr.setInterval(200); self._sb_tmr.timeout.connect(self._sb.tick); self._sb_tmr.start()

        if self._reader is not None:
            self._reader.frame_received.connect(self._on_frame)

    # ── Stylesheet ───────────────────────────────────────────

    def _stylesheet(self):
        return f"""
        QMainWindow, QWidget {{ background:{C.BG}; color:{C.WHITE};
            font-family:'Segoe UI','DejaVu Sans',sans-serif; }}
        QScrollArea {{ background:transparent; border:none; }}
        QScrollBar:vertical {{ background:{C.CARD2}; width:4px; border-radius:2px; }}
        QScrollBar::handle:vertical {{ background:{C.ACCENT_DK[C.MODE_SIGN]};
            border-radius:2px; min-height:16px; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
        """

    # ── UI ───────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        lay = QVBoxLayout(root); lay.setContentsMargins(8, 6, 8, 6); lay.setSpacing(5)

        lay.addWidget(self._build_header(), 0)
        body = QHBoxLayout(); body.setSpacing(6)
        body.addWidget(self._build_left(), 33)
        body.addWidget(self._build_right(), 67)
        lay.addLayout(body, 1)

    def _build_header(self):
        w = QWidget(); w.setFixedHeight(28)
        l = QHBoxLayout(w); l.setContentsMargins(4, 0, 4, 0); l.setSpacing(0)
        logo = QLabel("⬡  GESTIX  <span style='font-weight:400;color:#888;font-size:10px;'>2.0</span>")
        logo.setTextFormat(Qt.RichText)
        logo.setStyleSheet(f"color:{C.WHITE}; font-size:14px; font-weight:700; letter-spacing:3px;")
        l.addWidget(logo); l.addStretch()
        self._toggle = ModeToggle(); self._toggle.mode_changed.connect(self._switch_mode)
        l.addWidget(self._toggle); l.addSpacing(10)
        # Status
        self._status_dot = QLabel("●"); self._status_dot.setStyleSheet(f"color:{C.ACCENT[C.MODE_SIGN]}; font-size:7px;")
        self._status_txt = QLabel("semne"); self._status_txt.setStyleSheet(f"color:{C.WHITE}; font-size:10px;")
        l.addWidget(self._status_dot); l.addWidget(self._status_txt); l.addSpacing(8)
        self._clock = QLabel(); self._clock.setStyleSheet(f"color:{C.WHITE}; font-size:13px; font-family:monospace; font-weight:600;")
        l.addWidget(self._clock)
        return w

    def _build_left(self):
        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_sign_panel())
        self._stack.addWidget(self._build_stt_panel())
        return self._stack

    def _build_sign_panel(self):
        w = QWidget(); l = QVBoxLayout(w); l.setContentsMargins(0,0,0,0); l.setSpacing(5)
        card = self._card(); cl = card.layout()
        self._card_title(cl, "01", "Semn detectat", C.ACCENT[C.MODE_SIGN])
        box = QWidget(); box.setStyleSheet("background:#0b0b0b; border-radius:7px;")
        bl = QVBoxLayout(box); bl.setContentsMargins(6,6,6,6)
        self._sign_lbl = QLabel("—"); self._sign_lbl.setAlignment(Qt.AlignCenter)
        self._sign_lbl.setStyleSheet(f"color:{C.WHITE}; font-size:72px; font-weight:900; font-family:'Georgia',serif; background:transparent;")
        bl.addWidget(self._sign_lbl); cl.addWidget(box, 1)
        cr = QHBoxLayout()
        lbl = QLabel("confidence"); lbl.setStyleSheet(f"color:{C.GRAY}; font-size:10px;")
        cr.addWidget(lbl); cr.addStretch()
        self._conf_lbl = QLabel("—")
        self._conf_lbl.setStyleSheet(f"background:{C.GRAY2}; color:{C.GRAY}; font-size:10px; font-weight:bold; border-radius:4px; padding:2px 8px;")
        cr.addWidget(self._conf_lbl); cl.addLayout(cr)
        l.addWidget(card, 1)
        # Mod activ
        self._mode_lbl = QLabel("Mod: STATIC")
        self._mode_lbl.setStyleSheet(f"color:{C.GRAY}; font-size:9px;")
        l.addWidget(self._mode_lbl)
        return w

    def _build_stt_panel(self):
        w = QWidget(); l = QVBoxLayout(w); l.setContentsMargins(0,0,0,0); l.setSpacing(5)
        card = self._card(); cl = card.layout()
        self._card_title(cl, "01", "Voce detectată", C.ACCENT[C.MODE_STT])
        box = QWidget(); box.setStyleSheet("background:#0b0b0b; border-radius:7px;")
        bl = QVBoxLayout(box); bl.setContentsMargins(8,8,8,8); bl.setSpacing(6)
        mic = QLabel("🎙"); mic.setAlignment(Qt.AlignCenter); mic.setStyleSheet("font-size:34px; background:transparent;")
        bl.addWidget(mic)
        self._wave = WaveformBars(14); bl.addWidget(self._wave)
        cl.addWidget(box)
        self._stt_status = QLabel("inactiv")
        self._stt_status.setAlignment(Qt.AlignCenter)
        self._stt_status.setStyleSheet(f"color:{C.GRAY}; font-size:10px; font-style:italic;")
        cl.addWidget(self._stt_status)
        l.addWidget(card, 1)
        return w

    def _build_right(self):
        card = self._card(); cl = card.layout(); cl.setSpacing(6)
        tr = QHBoxLayout()
        title = QLabel("Conversație")
        title.setStyleSheet(f"color:{C.WHITE}; font-size:20px; font-weight:700; font-family:'Georgia',serif;")
        tr.addWidget(title); tr.addStretch()
        btn_clear = QPushButton("Curăță")
        btn_clear.setStyleSheet(f"background:transparent; color:{C.GRAY}; font-size:10px; border:1px solid {C.GRAY2}; border-radius:9px; padding:2px 8px;")
        btn_clear.setCursor(Qt.PointingHandCursor); btn_clear.clicked.connect(self._clear_chat)
        tr.addWidget(btn_clear); cl.addLayout(tr)
        # Chat
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._chat_w = QWidget(); self._chat_w.setStyleSheet("background:transparent;")
        self._chat_l = QVBoxLayout(self._chat_w)
        self._chat_l.setContentsMargins(0,2,4,2); self._chat_l.setSpacing(8); self._chat_l.addStretch()
        scroll.setWidget(self._chat_w); self._scroll = scroll
        cl.addWidget(scroll, 1)
        # Sentence preview
        sw = QWidget(); sw.setStyleSheet(f"background:{C.CARD2}; border-radius:12px; border:1px solid {C.GRAY2};")
        sl = QHBoxLayout(sw); sl.setContentsMargins(12,6,12,6); sl.setSpacing(8)
        tag = QLabel("◉"); tag.setStyleSheet(f"color:{C.ACCENT[C.MODE_SIGN]}; font-size:12px;")
        sl.addWidget(tag)
        self._preview = QLabel("(fă semne...)")
        self._preview.setStyleSheet(f"color:{C.GRAY}; font-size:13px; font-style:italic; font-family:'Georgia',serif;")
        sl.addWidget(self._preview, 1)
        btn_send = QPushButton("✉")
        btn_send.setStyleSheet(f"background:{C.ACCENT[C.MODE_SIGN]}; color:white; font-size:12px; font-weight:700; border-radius:9px; padding:3px 12px;")
        btn_send.setCursor(Qt.PointingHandCursor); btn_send.clicked.connect(self._sb.flush_sentence)
        sl.addWidget(btn_send)
        cl.addWidget(sw)
        return card

    # ── Helpers UI ───────────────────────────────────────────

    def _card(self):
        w = QWidget(); w.setStyleSheet(f"background:{C.CARD}; border-radius:9px;")
        l = QVBoxLayout(w); l.setContentsMargins(10,8,10,8); l.setSpacing(5)
        return w

    def _card_title(self, layout, num, title, accent):
        row = QHBoxLayout(); row.setSpacing(6)
        n = QLabel(num); n.setStyleSheet(f"color:{accent}; font-size:9px; font-weight:bold; border:1px solid {accent}; border-radius:3px; padding:1px 5px;")
        row.addWidget(n)
        t = QLabel(title); t.setStyleSheet(f"color:{C.WHITE}; font-size:12px; font-weight:700;")
        row.addWidget(t); row.addStretch(); layout.addLayout(row)

    def _tick(self):
        self._clock.setText(datetime.now().strftime("%H:%M:%S"))

    # ── Încărcare modele ─────────────────────────────────────

    def _load_models(self):
        if C.MODEL_STATIC_PATH:
            try:
                self._inf_static.load(C.MODEL_STATIC_PATH)
                self._buf_static = deque(maxlen=self._inf_static.window)
                print(f"[ML] Static încărcat: {self._inf_static.classes}")
            except Exception as exc:
                print(f"[ML] Static eșuat: {exc}")
        if C.MODEL_MOTION_PATH:
            try:
                self._inf_motion.load(C.MODEL_MOTION_PATH)
                self._buf_motion = deque(maxlen=self._inf_motion.window)
                print(f"[ML] Motion încărcat: {self._inf_motion.classes}")
            except Exception as exc:
                print(f"[ML] Motion eșuat: {exc}")

    # ── Switch mod ───────────────────────────────────────────

    def _switch_mode(self, mode):
        self._mode = mode
        accent = C.ACCENT[mode]
        self._stack.setCurrentIndex(0 if mode == C.MODE_SIGN else 1)
        self._status_dot.setStyleSheet(f"color:{accent}; font-size:7px;")
        self._status_txt.setText("semne" if mode == C.MODE_SIGN else "stt")
        if mode == C.MODE_SIGN:
            if self._stt:
                self._stt.stop(); self._stt = None
            self._wave_tmr.stop()
        else:
            self._wave_tmr.start(50)
            if self._stt is None:
                self._stt = STTWorker(C.WHISPER_MODEL, C.WHISPER_LANG, C.CHUNK_SECONDS)
                self._stt.status.connect(self._stt_status.setText)
                self._stt.final.connect(self._on_stt_final)
                self._stt.level.connect(self._wave.set_level)
                self._stt.start()

    # ── Inferență din frame-uri ──────────────────────────────

    def _on_frame(self, ts_ms: int, sensors: list):
        if self._mode != C.MODE_SIGN or len(sensors) != 21:
            return
        # Gyro magnitude → decide static/motion
        gm = math.sqrt(sum(sensors[i]**2 for i in GYRO_IDX))
        new_mode = "motion" if gm > C.GYRO_MOTION_THRESHOLD else "static"
        if new_mode != self._active_mode:
            self._active_mode = new_mode
            self._probs_ema = None; self._recent.clear()
            self._mode_lbl.setText(f"Mod: {new_mode.upper()}")

        inf = self._inf_motion if self._active_mode == "motion" else self._inf_static
        buf = self._buf_motion if self._active_mode == "motion" else self._buf_static
        if not inf.loaded:
            return

        buf.append(sensors)
        if len(buf) < inf.window:
            return
        self._stride += 1
        if self._stride < C.STRIDE_FRAMES:
            return
        self._stride = 0

        try:
            frames = np.array(buf, dtype=np.float32)
            # predict cu probabilități pentru smoothing
            pred, conf = inf.predict(frames)
        except Exception:
            return

        # Display
        disp = format_class_name(pred)
        self._sign_lbl.setText(disp if len(disp) <= 6 else disp[:5] + "…")
        self._conf_lbl.setText(f"{conf:.2f}")
        bg = C.ACCENT[C.MODE_SIGN] if conf >= C.CONFIDENCE_THRESHOLD else C.GRAY2
        self._conf_lbl.setStyleSheet(f"background:{bg}; color:white; font-size:10px; font-weight:bold; border-radius:4px; padding:2px 8px;")

        if conf < C.CONFIDENCE_THRESHOLD:
            return
        # Votare stabilitate
        self._recent.append(pred)
        if len(self._recent) >= STABILITY_FRAMES:
            last = list(self._recent)[-STABILITY_FRAMES:]
            if all(c == pred for c in last):
                self._sb.add_sign(pred, is_static=(self._active_mode == "static"))

    # ── STT callback ─────────────────────────────────────────

    def _on_stt_final(self, txt: str):
        if txt:
            self._add_bubble("stt", txt, on_right=False, accent=C.ACCENT[C.MODE_STT])

    # ── Sentence builder callbacks ───────────────────────────

    def _on_sentence_preview(self, text: str):
        if text:
            self._preview.setText(text + " │")
            self._preview.setStyleSheet(f"color:{C.WHITE}; font-size:14px; font-family:'Georgia',serif;")
        else:
            self._preview.setText("(fă semne...)")
            self._preview.setStyleSheet(f"color:{C.GRAY}; font-size:13px; font-style:italic; font-family:'Georgia',serif;")

    def _on_sentence_complete(self, text: str):
        self._add_bubble("semn", text, on_right=True, accent=C.ACCENT[C.MODE_SIGN])

    # ── Chat ─────────────────────────────────────────────────

    def _add_bubble(self, source, text, on_right, accent):
        w = QWidget(); w.setStyleSheet("background:transparent;")
        wl = QVBoxLayout(w); wl.setContentsMargins(0,0,0,0); wl.setSpacing(3)
        hr = QHBoxLayout(); hr.setSpacing(4)
        badge = QLabel(source); badge.setStyleSheet(f"background:{accent}; color:white; font-size:8px; font-weight:bold; border-radius:3px; padding:1px 6px;")
        ts = QLabel(datetime.now().strftime("%H:%M:%S")); ts.setStyleSheet(f"color:{C.GRAY}; font-size:8px;")
        if on_right:
            hr.addStretch(); hr.addWidget(ts); hr.addWidget(badge)
        else:
            hr.addWidget(badge); hr.addWidget(ts); hr.addStretch()
        wl.addLayout(hr)
        br = QHBoxLayout()
        bubble = QLabel(text); bubble.setWordWrap(True); bubble.setMaximumWidth(380)
        radius = "border-bottom-right-radius:3px;" if on_right else "border-bottom-left-radius:3px;"
        bubble.setStyleSheet(f"background:{accent}; color:white; font-size:13px; border-radius:14px; {radius} padding:9px 15px;")
        if on_right:
            br.addStretch(); br.addWidget(bubble)
        else:
            br.addWidget(bubble); br.addStretch()
        wl.addLayout(br)
        self._chat_l.insertWidget(self._chat_l.count() - 1, w)
        QTimer.singleShot(50, lambda: self._scroll.verticalScrollBar().setValue(
            self._scroll.verticalScrollBar().maximum()))

    def _clear_chat(self):
        while self._chat_l.count() > 1:
            item = self._chat_l.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._sb.clear()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

    def closeEvent(self, event):
        if self._stt:
            self._stt.stop()
        if self._reader:
            self._reader.stop()
        super().closeEvent(event)
