# Gestix 2.0 — Interfața Raspberry Pi

Aplicația finală care rulează pe **Raspberry Pi 4B** cu ecran 7" touchscreen.
Traduce limbaj semne → text (cu model ML) și voce → text (Whisper STT),
afișând o conversație în timp real.

---

## Structură

```
gestix_rpi/
├── main.py                  # Entry point (pornește fullscreen)
├── config.py                # TOATE setările (port, model, culori)
├── requirements_rpi.txt
├── core/
│   ├── serial_reader.py     # Citire de la hub ESP32-C3 (21 senzori mâna dreaptă)
│   ├── sign_inference.py    # Model ML .tflite (hook — pui calea în config)
│   ├── sentence_builder.py  # Acumulează semne → propoziție
│   └── stt_worker.py        # Whisper STT (faster-whisper)
└── ui/
    └── interface.py         # Interfața Gestix (QMainWindow fullscreen)
```

---

## Instalare pe Raspberry Pi

### 1. Sistem
```bash
sudo apt update
sudo apt install python3-pyqt5 libportaudio2 python3-numpy python3-serial
```

### 2. Python
```bash
cd gestix_rpi
pip install -r requirements_rpi.txt
```

> **Notă LSTM**: dacă modelul tău MOTION are LSTM (Flex ops), `tflite-runtime`
> NU e suficient — instalează `tensorflow-aarch64`. Modelele `dense_snapshot`
> și `cnn1d` merg cu `tflite-runtime` simplu.

---

## Configurare

Editează **`config.py`**:

```python
# Serial — None = auto-detect (caută ttyUSB/ttyACM)
SERIAL_PORT = None              # sau "/dev/ttyUSB0"

# Modele ML — pune căile tale aici
MODEL_STATIC_PATH = "/home/pi/gestix/models/gestix_static.tflite"
MODEL_MOTION_PATH = "/home/pi/gestix/models/gestix_motion.tflite"

FULLSCREEN = True              # True pe RPi, False la dezvoltare pe PC
```

Lângă fiecare model, copiază și `{model}.labels.json` + `{model}_preprocessor.json`
(generate automat de Gestix Trainer la salvare).

---

## Rulare

```bash
python3 main.py
```

- **ESC** = ieșire din fullscreen
- Toggle **SEMNE / STT** sus în dreapta
- Buton **✉** = trimite propoziția acum (fără să aștepți pauza)

---

## Cum funcționează

### Mod SEMNE
1. Hub-ul trimite 100 Hz date de la mănuși
2. Se folosesc DOAR cei 21 senzori ai mâinii drepte
3. Gyro magnitude decide: mână pe loc → **STATIC** (litere), mână în mișcare → **MOTION** (cuvinte)
4. Modelul prezice → semnele se acumulează în propoziție
5. Pauză 5s sau buton ✉ → propoziția apare ca **bulă roz în dreapta**

### Mod STT
1. Microfonul ascultă
2. Whisper transcrie offline (română)
3. Textul apare ca **bulă albastră în stânga**

---

## Pentru a conecta modelul ML (Lucian)

În `config.py` pune calea către `.tflite`. Aplicația încarcă automat:
- modelul
- `.labels.json` (clasele)
- `_preprocessor.json` (normalizarea — adaptive sau imu_only)

`SignInference` detectează automat tipul (Hall+IMU sau imu_only) din preprocessor.

---

## Autostart la boot (opțional)

Pentru pornire automată pe RPi, adaugă în `/etc/xdg/autostart/gestix.desktop`:
```ini
[Desktop Entry]
Type=Application
Name=Gestix
Exec=python3 /home/pi/gestix_rpi/main.py
```

---

## Echipă

Balcan Radu (software) · Bandula Lucian (hardware) · Mentor Mîndroiu Lucian
Liceul „Grigore Moișil" Tulcea — ONCS 2026
