VideoAnalyzer Tools
===================

FFMPEG Analyzer

This repository contains a small ffprobe-based analyzer and a Qt GUI viewer that shows decoded video frames along with PTS/DTS metadata and a short audio waveform corresponding to each frame.

Requirements
------------
- ffmpeg / ffprobe (available on PATH)
- Python packages (see `requirements.txt`)

GUI usage
---------
Run the GUI viewer for a file:

```bash
python -m ffmpeg_analyser --gui <path-to-video>
```

Or run the GUI module directly:

```bash
python gui_analyser.py <path-to-video>
```

Generate AV-sync test clip
-------------------------
I included small helper scripts to generate a test clip (white flash + audible beep). They create an MP4, WAV and a waveform PNG for quick inspection.

From WSL / bash:

```bash
./tools/generate_sync_clip.sh [out.mp4] [duration_seconds]
# example (5s):
./tools/generate_sync_clip.sh out_sync_test_loud.mp4 5
```

From PowerShell:

```powershell
./tools/generate_sync_clip.ps1 -Out out_sync_test_loud.mp4 -Duration 5
```

The script will print the created filenames. Play the file with `ffplay` to verify the flash+beep alignment.

Safe generator (avoids PowerShell quoting issues)
------------------------------------------------
If you had trouble running the earlier scripts from PowerShell, use the "safe" generators which write the filter to a temporary file and call ffmpeg with `-filter_complex_script` to avoid quoting problems.

From WSL / bash:

```bash
./tools/generate_sync_clip_safe.sh out_sync_test_safe.mkv 5
```

From PowerShell:

```powershell
./tools/generate_sync_clip_safe.ps1 -Out out_sync_test_safe.mkv -Duration 5
```

These create an MKV with PCM audio (more portable for debugging), plus a WAV and waveform PNG.

Additional GUI setup
--------------------
The `video_analyzer_gui.py` tool provides a Tkinter-based GUI for MPEG-TS analysis and requires the following Python dependencies:

- PySide6>=6.0
- av
- numpy
- matplotlib
- Pillow
- opencv-python>=4.5

Optional extras:
- `pytesseract` for OCR support (requires system-installed Tesseract)
- NDI support requires the NewTek NDI SDK and a compatible Python binding. See `NDI_README.md` for NDI setup instructions.

Quick setup
-----------
1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

### Run the GUI

```bash
python3 video_analyzer_gui.py
```

### Run the CLI analyser

```bash
python3 video_analyzer.py input.ts --json
```

Local dependencies used by the GUI
----------------------------------
- `video_analyzer.py`
- `mp4_parser.py`
- `scte35_validator.py`
- `hevc_parser.py`
- `buffer_analyzer.py`
- `ndi_streamer.py` (optional)
- `ndi_analysis.py` (optional)
- `ndi_recorder.py` (optional)
