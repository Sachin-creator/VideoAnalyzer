#!/usr/bin/env python3
"""Simple GUI viewer: shows decoded video frames with PTS/DTS and corresponding audio waveform.

Dependencies: PySide6, av (PyAV), numpy, matplotlib

This module exposes launch_gui(path) which starts a Qt application window.
"""
from __future__ import annotations
import json
import subprocess
import sys
from typing import List, Dict, Any, Optional

try:
    from PySide6.QtWidgets import (
        QApplication,
        QWidget,
        QLabel,
        QPushButton,
        QHBoxLayout,
        QVBoxLayout,
        QSlider,
        QSizePolicy,
        QScrollArea,
    )
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QImage, QPixmap
except Exception as e:
    raise ImportError("PySide6 is required for the GUI: pip install PySide6") from e

try:
    import av
except Exception as e:
    raise ImportError("PyAV (av) is required for decoding: pip install av") from e

import numpy as np

# Matplotlib for waveform
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
except Exception:
    raise ImportError("matplotlib is required for waveform drawing: pip install matplotlib")


def _ffprobe_video_frames(path: str) -> List[Dict[str, Any]]:
    """Return ffprobe JSON frames for the video stream only (with packet times)."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-of",
        "json",
        "-show_frames",
        "-select_streams",
        "v",
        "-show_entries",
        "frame=pkt_pts_time,pkt_dts_time,pts_time,dts_time",
        path,
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, encoding="utf-8")
    data = json.loads(out)
    return data.get("frames", [])


class WaveformCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=1.5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = fig.add_subplot(111)
        self.ax.set_ylim(-1.0, 1.0)
        self.ax.set_yticks([])
        self.ax.set_xlabel("Time (s)")
        super().__init__(fig)

    def plot_segment(self, samples: np.ndarray, sample_rate: int, start_time: float = 0.0, marker_time: Optional[float] = None):
        self.ax.clear()
        if samples.size == 0:
            self.ax.set_ylim(-1.0, 1.0)
            self.draw()
            return
        t = np.linspace(start_time, start_time + samples.shape[-1] / sample_rate, samples.shape[-1])
        self.ax.plot(t, samples, linewidth=0.6)
        if t[0] == t[-1]:
            self.ax.set_xlim(t[0] - 0.01, t[0] + 0.01)
        else:
            self.ax.set_xlim(t[0], t[-1])
        self.ax.set_ylim(-1.0, 1.0)
        self.ax.set_yticks([])
        self.ax.set_xlabel("Time (s)")
        # optional marker (absolute seconds) — draw if inside window
        if marker_time is not None:
            try:
                if marker_time >= t[0] and marker_time <= t[-1]:
                    self.ax.axvline(marker_time, color="r", linewidth=1.0)
            except Exception:
                pass
        self.draw()

    def plot_full(self, samples: np.ndarray, sample_rate: int, marker_time: Optional[float] = None):
        """Plot the full waveform and optionally draw a vertical marker at marker_time (seconds)."""
        self.ax.clear()
        if samples.size == 0:
            self.ax.set_ylim(-1.0, 1.0)
            self.draw()
            return
        # downsample for plotting if very long
        total = samples.shape[-1]
        max_points = 20000
        if total > max_points:
            # take min/max per block to preserve peaks
            block = int(np.ceil(total / max_points))
            trimmed = samples[: block * max_points]
            reshaped = trimmed.reshape(-1, block)
            mins = reshaped.min(axis=1)
            maxs = reshaped.max(axis=1)
            t = np.linspace(0.0, total / sample_rate, mins.shape[0] * 2)
            data = np.empty((mins.shape[0] * 2,))
            data[0::2] = mins
            data[1::2] = maxs
            self.ax.plot(t, data, linewidth=0.4)
        else:
            t = np.linspace(0.0, total / sample_rate, total)
            self.ax.plot(t, samples, linewidth=0.6)

        if marker_time is not None:
            self.ax.axvline(marker_time, color="r", linewidth=1.0)

        if total == 1:
            self.ax.set_xlim(0, 1.0 / sample_rate)
        else:
            self.ax.set_xlim(0, total / sample_rate)
        self.ax.set_ylim(-1.0, 1.0)
        self.ax.set_yticks([])
        self.ax.set_xlabel("Time (s)")
        self.draw()


class VideoViewer(QWidget):
    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path = path

        # ffprobe metadata for PTS/DTS
        try:
            self.frame_infos = _ffprobe_video_frames(path)
        except Exception as e:
            print("ffprobe failed to get frame metadata:", e, file=sys.stderr)
            self.frame_infos = []

        # open separate containers for video and audio so demuxing one doesn't consume the other's packets
        try:
            self.video_container = av.open(path)
            self.audio_container = av.open(path)
        except Exception as e:
            print("Failed to open media with PyAV:", e, file=sys.stderr)
            raise

        self.video_stream = next((s for s in self.video_container.streams if s.type == "video"), None)
        self.audio_stream = next((s for s in self.audio_container.streams if s.type == "audio"), None)

        # debug: print discovered streams
        try:
            print("Discovered streams:")
            for s in self.video_container.streams:
                print(f"  stream idx={s.index} type={s.type} codec={getattr(s.codec_context,'name',None)}")
        except Exception:
            pass

        # decode audio fully to buffer (from its own container)
        self.audio_samples = np.zeros((0,), dtype=np.float32)
        self.audio_rate = 44100
        if self.audio_stream is not None:
            try:
                self.audio_rate = int(self.audio_stream.rate)
            except Exception:
                self.audio_rate = 44100
            self.audio_samples = self._decode_audio_to_numpy()

        # GUI elements
        self.video_label = QLabel("(no frame)")
        self.video_label.setFixedSize(1280, 720)  # Prevents dynamic resizing and Wayland crash
        self.video_label.setAlignment(Qt.AlignCenter)

        self.info_label = QLabel("")

        self.wave_canvas = WaveformCanvas(self, width=6, height=1.5)

        # thumbnail strip: show multiple frames with small waveforms
        self.window_size = 10
        # thumb_items stores (wrapper_widget, label, info_label, canvas)
        self.thumb_items = []  # list of (wrapper, label, info_label, canvas)
        self.thumb_container = QWidget()
        self.thumb_layout = QHBoxLayout(self.thumb_container)
        for i in range(self.window_size):
            tlabel = QLabel("(no frame)")
            tlabel.setFixedSize(160, 90)
            tlabel.setAlignment(Qt.AlignCenter)
            tinfo = QLabel("")
            twave = WaveformCanvas(self, width=2, height=1)
            v = QVBoxLayout()
            v.addWidget(tlabel)
            v.addWidget(tinfo)
            v.addWidget(twave)
            wrapper = QWidget()
            wrapper.setLayout(v)
            self.thumb_layout.addWidget(wrapper)
            self.thumb_items.append((wrapper, tlabel, tinfo, twave))

        self.thumb_scroll = QScrollArea()
        self.thumb_scroll.setWidgetResizable(True)
        self.thumb_scroll.setWidget(self.thumb_container)

        # controls
        self.play_btn = QPushButton("Play")
        self.prev_btn = QPushButton("Prev")
        self.next_btn = QPushButton("Next")
        self.slider = QSlider(Qt.Horizontal)

        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(self.play_btn)
        ctrl_layout.addWidget(self.prev_btn)
        ctrl_layout.addWidget(self.next_btn)
        ctrl_layout.addWidget(self.slider)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label, stretch=6)
        layout.addWidget(self.info_label)
        layout.addWidget(self.wave_canvas, stretch=1)
        layout.addWidget(self.thumb_scroll, stretch=2)
        layout.addLayout(ctrl_layout)
        self.setLayout(layout)

        # state
        self.frames = []
        self._load_video_frames()
        self.index = 0

        self.slider.setMinimum(0)
        self.slider.setMaximum(max(0, len(self.frames) - 1))
        self.slider.valueChanged.connect(self.on_slider)

        self.playing = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_next)

        self.play_btn.clicked.connect(self.on_play)
        self.prev_btn.clicked.connect(self.on_prev)
        self.next_btn.clicked.connect(self.on_next)

        self.setWindowTitle(f"Video Analyzer: {path}")
        # Defer first display until the widget is shown (ensures correct widget sizes)
        QTimer.singleShot(0, lambda: self._on_ready())

    def _on_ready(self):
        # load debug info and display first frame if available
        print(f"Loaded video frames: {len(self.frames)}")
        print(f"Loaded ffprobe frame infos: {len(self.frame_infos)}")
        print(f"Audio samples: {self.audio_samples.shape[0]} @ {self.audio_rate} Hz")
        if self.frames:
            self.display_frame(0)
        else:
            self.info_label.setText("No decoded video frames found — check PyAV / ffmpeg or try a short test file.")

    def _decode_audio_to_numpy(self) -> np.ndarray:
        """Decode audio stream to numpy array with shape (samples,) mono mix."""
        buffers = []
        try:
            # demux packets and decode only audio packets to avoid stream index issues
            if self.audio_stream is None:
                return np.zeros((0,), dtype=np.float32)
            for packet in self.audio_container.demux():
                if packet.stream.type != 'audio':
                    continue
                for frame in packet.decode():
                    arr = frame.to_ndarray()  # shape (channels, samples)
                    # convert to float32 in [-1,1]
                    if np.issubdtype(arr.dtype, np.integer):
                        maxv = float(2 ** (8 * arr.dtype.itemsize - 1))
                        f = arr.astype(np.float32) / maxv
                    else:
                        f = arr.astype(np.float32)
                    # mixdown to mono
                    if f.ndim == 2:
                        f = np.mean(f, axis=0)
                    buffers.append(f)
        except Exception:
            import traceback
            traceback.print_exc()
        if buffers:
            return np.concatenate(buffers, axis=0)
        return np.zeros((0,), dtype=np.float32)

    def _load_video_frames(self):
        if self.video_stream is None:
            return
        try:
            # prefer direct decode to avoid demux/packet ordering issues
            for frame in self.video_container.decode(video=self.video_stream.index):
                # convert to RGB numpy array
                arr = frame.to_ndarray(format="rgb24")
                # store tuple (frame, pixel_array) because av.VideoFrame is a C-extension object
                # and doesn't allow attaching arbitrary attributes
                self.frames.append((frame, arr))
        except Exception:
            import traceback
            traceback.print_exc()

    def display_frame(self, idx: int):
        if idx < 0 or idx >= len(self.frames):
            return
        self.index = idx
        item = self.frames[idx]
        if isinstance(item, tuple) or isinstance(item, list):
            frame, arr = item
        else:
            frame = item
            arr = getattr(frame, "_pixel_array", None)
        if arr is None:
            return
        h, w, ch = arr.shape
        bytes_per_line = ch * w
        qimg = QImage(arr.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(pix)

        # fetch metadata
        info = {"PTS": "N/A", "DTS": "N/A"}
        if idx < len(self.frame_infos):
            fi = self.frame_infos[idx]
            info["PTS"] = fi.get("pkt_pts_time") or fi.get("pts_time") or "N/A"
            info["DTS"] = fi.get("pkt_dts_time") or fi.get("dts_time") or "N/A"
        else:
            # fallback to frame.pts if available
            try:
                if frame.pts is not None and frame.time_base is not None:
                    info["PTS"] = f"{float(frame.pts * frame.time_base):.6f}"
            except Exception:
                pass

        self.info_label.setText(f"Frame {idx+1}/{len(self.frames)} — PTS: {info['PTS']}   DTS: {info['DTS']}")

        # waveform: show short window around PTS
        try:
            pts = None
            if info["PTS"] != "N/A":
                pts = float(info["PTS"])
        except Exception:
            pts = None

        if pts is not None and self.audio_samples.size > 0:
            win_s = 0.2
            center = int(pts * self.audio_rate)
            half = int(win_s * self.audio_rate / 2)
            start = max(0, center - half)
            end = min(self.audio_samples.shape[0], center + half)
            seg = self.audio_samples[start:end]
            if seg.size == 0:
                seg = np.zeros((1,), dtype=np.float32)
            # normalize for plotting
            if np.max(np.abs(seg)) > 0:
                seg = seg / np.max(np.abs(seg))
            # plot main waveform segment and draw marker at exact PTS
            self.wave_canvas.plot_segment(seg, self.audio_rate, start_time=start / self.audio_rate, marker_time=pts)
        else:
            self.wave_canvas.plot_segment(np.zeros((1,), dtype=np.float32), self.audio_rate, 0.0)

        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)

        # update thumbnail window centered on current frame
        try:
            self.update_thumbnails(idx)
        except Exception:
            pass

        # visually highlight currently selected main frame
        try:
            # red border on main display
            self.video_label.setStyleSheet("border: 3px solid red;")
        except Exception:
            pass

    def get_pts_for_index(self, idx: int) -> Optional[float]:
        """Return PTS in seconds for a given frame index if available."""
        if idx < 0 or idx >= len(self.frames):
            return None
        # prefer ffprobe packet times
        if idx < len(self.frame_infos):
            fi = self.frame_infos[idx]
            pts_s = fi.get("pkt_pts_time") or fi.get("pts_time")
            try:
                return float(pts_s) if pts_s is not None and pts_s != "" else None
            except Exception:
                return None
        # fallback to frame.pts
        item = self.frames[idx]
        frame = item[0] if isinstance(item, (tuple, list)) else item
        try:
            if frame.pts is not None and frame.time_base is not None:
                return float(frame.pts * frame.time_base)
        except Exception:
            return None
        return None

    def get_audio_segment_for_pts(self, pts: float, win_s: float = 0.2) -> np.ndarray:
        if pts is None or self.audio_samples.size == 0:
            return np.zeros((0,), dtype=np.float32)
        center = int(pts * self.audio_rate)
        half = int(win_s * self.audio_rate / 2)
        start = max(0, center - half)
        end = min(self.audio_samples.shape[0], center + half)
        seg = self.audio_samples[start:end]
        if seg.size == 0:
            return np.zeros((0,), dtype=np.float32)
        # normalize
        if np.max(np.abs(seg)) > 0:
            seg = seg / np.max(np.abs(seg))
        return seg

    def update_thumbnails(self, center_idx: int):
        n = len(self.frames)
        if n == 0:
            return
        half = self.window_size // 2
        start = max(0, center_idx - half)
        if start + self.window_size > n:
            start = max(0, n - self.window_size)
        for i in range(self.window_size):
            idx = start + i
            wrapper, tlabel, tinfo, twave = self.thumb_items[i]
            if idx < n:
                item = self.frames[idx]
                if isinstance(item, (tuple, list)):
                    frame, arr = item
                else:
                    frame = item
                    arr = getattr(frame, "_pixel_array", None)
                if arr is not None:
                    h, w, ch = arr.shape
                    qimg = QImage(arr.data.tobytes(), w, h, ch * w, QImage.Format_RGB888)
                    pix = QPixmap.fromImage(qimg).scaled(tlabel.width(), tlabel.height(), Qt.KeepAspectRatio)
                    tlabel.setPixmap(pix)
                else:
                    tlabel.setText("(no frame)")
                # pts/dts
                pts = self.get_pts_for_index(idx)
                if pts is not None:
                    tinfo.setText(f"{idx+1}\nPTS:{pts:.3f}")
                else:
                    tinfo.setText(f"{idx+1}\nPTS:N/A")
                # waveform around pts
                if pts is not None:
                    seg = self.get_audio_segment_for_pts(pts, win_s=0.2)
                    if seg.size > 0:
                        # draw a small marker at pts for the thumbnail waveform
                        twave.plot_segment(seg, self.audio_rate, start_time=max(0.0, pts - 0.1), marker_time=pts)
                    else:
                        twave.plot_segment(np.zeros((1,), dtype=np.float32), self.audio_rate, 0.0)
                else:
                    twave.plot_segment(np.zeros((1,), dtype=np.float32), self.audio_rate, 0.0)
                # highlight selected thumbnail
                try:
                    if idx == center_idx:
                        wrapper.setStyleSheet("border: 2px solid yellow;")
                    else:
                        wrapper.setStyleSheet("")
                except Exception:
                    pass
            else:
                tlabel.clear()
                tinfo.setText("")
                twave.plot_segment(np.zeros((1,), dtype=np.float32), self.audio_rate, 0.0)
                try:
                    wrapper.setStyleSheet("")
                except Exception:
                    pass

    def on_prev(self):
        self.display_frame(max(0, self.index - 1))

    def on_next(self):
        if self.index + 1 < len(self.frames):
            self.display_frame(self.index + 1)
        else:
            self.on_stop()

    def on_play(self):
        if not self.playing:
            # approximate frame interval by stream average rate
            fps = 25.0
            try:
                if self.video_stream is not None and self.video_stream.average_rate is not None:
                    fps = float(self.video_stream.average_rate)
            except Exception:
                pass
            interval = int(1000 / max(1.0, fps))
            self.timer.start(interval)
            self.play_btn.setText("Pause")
            self.playing = True
        else:
            self.on_stop()

    def on_stop(self):
        self.timer.stop()
        self.play_btn.setText("Play")
        self.playing = False

    def on_slider(self, val: int):
        self.display_frame(val)


def launch_gui(path: str):
    app = QApplication(sys.argv)
    viewer = VideoViewer(path)
    screen = app.primaryScreen()
    size = screen.availableGeometry()
    viewer.setGeometry(size)  # Set window to max available size
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: gui_analyser.py <video-file>")
        sys.exit(1)
    launch_gui(sys.argv[1])
