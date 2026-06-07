"""Simple NDI frame recorder using OpenCV VideoWriter.

Writes raw frames (numpy arrays) to an MP4 file. Requires `cv2`.
"""
from __future__ import annotations
from typing import Optional
import threading
import time

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None  # type: ignore
    np = None  # type: ignore


class RecorderError(RuntimeError):
    pass


class NDIRecorder:
    def __init__(self, path: str, fps: float = 25.0, fourcc: str = 'mp4v') -> None:
        if cv2 is None:
            raise ImportError('opencv-python is required for NDIRecorder')
        self.path = path
        self.fps = float(fps)
        self.fourcc = fourcc
        self._writer = None
        self._lock = threading.Lock()
        self._started = False

    def _init_writer(self, frame):
        h, w = frame.shape[0], frame.shape[1]
        fourcc = cv2.VideoWriter_fourcc(*self.fourcc)
        self._writer = cv2.VideoWriter(self.path, fourcc, self.fps, (w, h))
        if not self._writer.isOpened():
            raise RecorderError('Failed to open VideoWriter')
        self._started = True

    def write_frame(self, frame) -> None:
        if cv2 is None:
            raise ImportError('opencv-python is required for recording')
        with self._lock:
            if not self._started:
                self._init_writer(frame)
            # Ensure frame is BGR uint8
            try:
                arr = np.asarray(frame)
                if arr.dtype != np.uint8:
                    arr = arr.astype('uint8')
                # If grayscale convert to BGR
                if arr.ndim == 2:
                    arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                elif arr.ndim == 3 and arr.shape[2] == 4:
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
                self._writer.write(arr)
            except Exception as e:
                raise RecorderError(f'Failed to write frame: {e}')

    def close(self) -> None:
        with self._lock:
            try:
                if self._writer:
                    self._writer.release()
            finally:
                self._writer = None
                self._started = False
