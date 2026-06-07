# VideoAnalyzer

Small GUI and CLI tooling for analysing MPEG-TS, MP4/MOV and related media.

## Overview
This repository contains a Tkinter-based GUI (`video_analyzer_gui.py`) and a CLI analyser (`video_analyzer.py`) for inspecting transport streams, H.264/HEVC parsing, SCTE-35 validation and several related helpers (NDI support, buffer analysis, MP4 parsing).

## Quick setup
1. Create and activate a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

Notes:
- `av` is the PyAV (FFmpeg bindings) package and requires FFmpeg headers/system libs available on the host.
- NDI support requires the NewTek NDI SDK (not provided via PyPI). See `NDI_README.md` for instructions.
- `pytesseract` requires Tesseract OCR installed on the system if you intend to use OCR features.

## Run the GUI

```bash
python3 video_analyzer_gui.py
```

The GUI will let you open local files, NDI sources (if NDI SDK/bindings are installed) and display analysis results and graphs.

## Run CLI analyser

```bash
python3 video_analyzer.py input.ts --json
```

## Local dependencies (source files used by the GUI)
- `video_analyzer.py`
- `mp4_parser.py`
- `scte35_validator.py`
- `hevc_parser.py`
- `buffer_analyzer.py`
- `ndi_streamer.py` (optional)
- `ndi_analysis.py` (optional)
- `ndi_recorder.py` (optional)

## Branch
Changes were made on branch `gui/deps-list` which has been pushed to origin.

## Troubleshooting
- If `av` installation fails, ensure FFmpeg and its development headers are installed (platform-specific packages).
- If the GUI appears blank on WSL, ensure an X/Wayland/Tk display is available (use WSLg or an X server).

If you'd like, I can open a PR draft with this README or expand with examples/screenshots.
