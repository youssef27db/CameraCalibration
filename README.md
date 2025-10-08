# CameraCalibration

Dieses Projekt dient der Kamerakalibrierung mithilfe von Checkerboard-Bildern. Es wurde für wissenschaftliche Zwecke im Rahmen einer Bachelorarbeit entwickelt.

## Projektübersicht

Die Kalibrierung einer Kamera ist ein wichtiger Schritt, um Verzerrungen zu korrigieren und präzise Messungen in Computer-Vision-Anwendungen zu ermöglichen. Dieses Projekt umfasst:

* **Generierung von Checkerboard-Bildern**
  Erstellung synthetischer oder gespeicherter Checkerboard-Bilder für die Kalibrierung.

* **Kamerakalibrierung**
  Detektion der Checkerboard-Ecken und Berechnung der Kameramatrix sowie der Verzerrungskoeffizienten.

## Verzeichnisstruktur

```
CameraCalibration/
├── images/            # Checkerboard-Bilder für die Kalibrierung
├── calibration.py     # Skript zur Kamerakalibrierung
├── createImages.py    # Skript zur Erstellung von Checkerboard-Bildern
├── .gitattributes     # Git-Datei
└── README.md          # Projektübersicht
```

## Installation

1. Klone das Repository:

```bash
git clone https://github.com/youssef27db/CameraCalibration.git
cd CameraCalibration
```

2. Installiere die benötigten Python-Pakete (empfohlen: virtuelles Environment):

```bash
pip install opencv-python numpy
```

## Verwendung

### Checkerboard-Bilder erstellen

```bash
python createImages.py
```

Das Skript öffnet die Kamera, mit s speichert man Bilder im images/-Ordner, mit ESC beendet man das Programm.

### Kamera kalibrieren

```bash
python calibration.py
```

Das Skript erkennt die Checkerboard-Ecken in den Bildern und berechnet die Kameramatrix sowie die Verzerrungskoeffizienten. Ergebnisse werden in der Konsole ausgegeben.

## Lizenz

Dieses Projekt ist für wissenschaftliche Zwecke freigegeben. Für kommerzielle Nutzung bitte Rücksprache halten.
