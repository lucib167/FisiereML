"""
Sentence Builder — acumulează semnele într-o propoziție.

Reguli:
  - Literă STATIC      → concat la cuvântul curent (L-U-C-I → "luci")
  - Cuvânt MOTION      → flush cuvântul curent, adaugă cuvântul complet
  - IDLE > WORD_GAP    → flush cuvânt (separator între cuvinte)
  - IDLE > SENTENCE    → emite propoziția (callback on_sentence)

Independent de UI — testabil unitar.
"""

import time
from typing import Callable, Optional


def format_class_name(name: str) -> str:
    """'Ma_Numesc' → 'ma numesc'; 'A' → 'A' (litere izolate rămân majuscule)."""
    cleaned = name.strip()
    if not cleaned:
        return cleaned
    if len(cleaned) <= 2 and cleaned.isalpha():
        return cleaned.upper()
    return cleaned.replace("_", " ").lower()


# Clase neutre — ignorate complet
IGNORED = {"idle", "neutral", "rest", "none", "nimic", "_idle_",
           "idle_motion", "idle_static"}


def is_idle(class_name: str) -> bool:
    return class_name.lower().strip() in IGNORED


class SentenceBuilder:
    """
    Construiește propoziții din semne detectate.

    on_update(text)   — apelat când preview-ul se schimbă
    on_sentence(text) — apelat când propoziția e completă (gata de bulă)
    """

    def __init__(self, word_gap_sec: float = 2.0, sentence_end_sec: float = 5.0,
                 on_update: Optional[Callable[[str], None]] = None,
                 on_sentence: Optional[Callable[[str], None]] = None):
        self.word_gap = word_gap_sec
        self.sentence_end = sentence_end_sec
        self.on_update = on_update or (lambda t: None)
        self.on_sentence = on_sentence or (lambda t: None)

        self._word = ""
        self._sentence = ""
        self._last_sign_t = 0.0
        self._last_emit_class: Optional[str] = None

    # ── Adăugare semn ────────────────────────────────────────

    def add_sign(self, raw_class: str, is_static: bool):
        """Adaugă un semn detectat (deja trecut de threshold + smoothing)."""
        if is_idle(raw_class):
            # IDLE = reset state pentru repetare imediată
            self._last_emit_class = None
            return

        # Dedup pe clasă consecutivă
        if raw_class == self._last_emit_class:
            return

        display = format_class_name(raw_class)
        is_letter = is_static and len(display) <= 2 and display.isalpha()

        if is_letter:
            self._word += display.lower()
        else:
            self._flush_word()
            self._sentence = (self._sentence + " " + display).strip() \
                if self._sentence else display

        self._last_emit_class = raw_class
        self._last_sign_t = time.monotonic()
        self.on_update(self.preview())

    # ── Timing (apelat periodic, ex 200ms) ───────────────────

    def tick(self):
        """Verifică pauzele IDLE → flush cuvânt sau propoziție."""
        if self._last_sign_t == 0.0:
            return
        idle = time.monotonic() - self._last_sign_t

        if idle >= self.word_gap and self._word:
            self._flush_word()
            self.on_update(self.preview())

        if idle >= self.sentence_end and (self._sentence or self._word):
            self.flush_sentence()

    # ── Flush ────────────────────────────────────────────────

    def _flush_word(self):
        if not self._word:
            return
        self._sentence = (self._sentence + " " + self._word).strip() \
            if self._sentence else self._word
        self._word = ""

    def flush_sentence(self):
        """Trimite propoziția completă (manual sau auto)."""
        self._flush_word()
        s = self._sentence.strip()
        if not s:
            return
        formatted = s[0].upper() + s[1:]
        if not formatted.endswith((".", "!", "?")):
            formatted += "."
        self.on_sentence(formatted)
        self._sentence = ""
        self._word = ""
        self._last_emit_class = None
        self._last_sign_t = 0.0
        self.on_update(self.preview())

    def clear(self):
        self._sentence = ""
        self._word = ""
        self._last_emit_class = None
        self._last_sign_t = 0.0
        self.on_update(self.preview())

    def preview(self) -> str:
        full = self._sentence
        if self._word:
            full = (full + " " + self._word).strip()
        return full
