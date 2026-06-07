"""Minimal NDI receiver wrapper used by VideoAnalyzer.

This module provides a thin `NDIReceiver` class that attempts to use a
Python NDI binding if available. The NewTek NDI SDK (native libraries)
must be installed on the host for most bindings to work.

The API here is intentionally small and defensive: callers should
use `list_sources()` to inspect available NDI sources, and `start()` to
begin a background thread that calls `frame_callback(frame)` with
a NumPy image (H x W x C, BGR) when frames arrive.

If no Python NDI binding is present an ImportError with guidance is
raised. See NDI_README.md for installation notes.
"""
from __future__ import annotations
import threading
import time
from typing import Callable, List, Optional

try:
    import ndi  # type: ignore
    _HAS_NDI = True
except Exception:
    ndi = None  # type: ignore
    _HAS_NDI = False

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore


class NDIError(RuntimeError):
    pass


class NDIReceiver:
    """Simple NDI receiver wrapper.

    Note: The exact Python binding API varies between wrappers. This
    class uses a best-effort approach and documents explicit failures
    so callers can take corrective action.
    """

    def __init__(self) -> None:
        if not _HAS_NDI:
            raise ImportError("No Python NDI binding found. See NDI_README.md for install steps.")
        self._ndi = ndi
        self._recv = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def list_sources(self) -> List[str]:
        """Return a list of available NDI source names."""
        try:
            sources = self._ndi.find_sources()
        except Exception:
            # Some bindings expose a global `find_sources()` while others
            # may have different names; re-raise with guidance.
            raise NDIError("Failed to enumerate NDI sources using the installed binding.")
        out = []
        for s in sources:
            # Try common attributes
            name = getattr(s, 'name', None) or getattr(s, 'source_name', None) or str(s)
            out.append(name)
        return out

    def start(self, source_name: Optional[str] = None, frame_callback: Optional[Callable] = None) -> None:
        """Start receiving from `source_name` (or first available).

        `frame_callback` will be called as `frame_callback(ndarray)` where
        `ndarray` is a HxWxC NumPy array in BGR ordering when available.
        """
        if not _HAS_NDI:
            raise ImportError("No Python NDI binding found. See NDI_README.md for install steps.")

        sources = self._ndi.find_sources()
        if not sources:
            raise NDIError("No NDI sources discovered on the network")

        pick = None
        if source_name:
            for s in sources:
                name = getattr(s, 'name', None) or getattr(s, 'source_name', None) or str(s)
                if name == source_name:
                    pick = s
                    break
            if pick is None:
                raise NDIError(f"NDI source '{source_name}' not found")
        else:
            pick = sources[0]

        # Create a receiver using common binding patterns. If the binding
        # uses a different constructor this may raise; surface as NDIError.
        try:
            self._recv = self._ndi.Receiver(pick)
        except Exception:
            raise NDIError("Failed to create NDI receiver with the installed binding")

        self._running = True

        def _loop():
            while self._running:
                try:
                    # Many bindings expose `capture_video(timeout_ms)` or similar
                    frame = None
                    if hasattr(self._recv, 'capture_video'):
                        frame = self._recv.capture_video(5000)
                    elif hasattr(self._recv, 'receive_video'):
                        frame = self._recv.receive_video(5000)
                    else:
                        raise NDIError('Installed NDI binding does not expose expected receive API')

                    if not frame:
                        continue

                    # Attempt to get an ndarray. Common wrappers provide
                    # `.to_ndarray()` or expose `.data`/`.video_buffer`.
                    img = None
                    if hasattr(frame, 'to_ndarray'):
                        img = frame.to_ndarray()
                    elif hasattr(frame, 'data'):
                        img = frame.data
                    elif hasattr(frame, 'video_buffer'):
                        img = frame.video_buffer

                    if img is None:
                        # Skip unsupported frame object
                        continue

                    if np is not None and not isinstance(img, np.ndarray):
                        try:
                            img = np.asarray(img)
                        except Exception:
                            pass

                    if frame_callback:
                        try:
                            frame_callback(img)
                        except Exception:
                            # Caller's callback must handle exceptions
                            pass

                except Exception:
                    # On transient errors sleep a bit and continue
                    time.sleep(0.1)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        try:
            if self._recv and hasattr(self._recv, 'close'):
                self._recv.close()
        except Exception:
            pass
