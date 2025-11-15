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
.\tools\generate_sync_clip.ps1 -Out out_sync_test_loud.mp4 -Duration 5
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
.\tools\generate_sync_clip_safe.ps1 -Out out_sync_test_safe.mkv -Duration 5
```

These create an MKV with PCM audio (more portable for debugging), plus a WAV and waveform PNG.



Notes
-----
The GUI uses PySide6 for the UI, PyAV for decoding, numpy and matplotlib for waveform plotting. If you don't need the GUI, the original CLI analysis (prints PTS/DTS via ffprobe) remains available.
