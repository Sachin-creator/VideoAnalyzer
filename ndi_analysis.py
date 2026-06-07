"""Per-frame analysis utilities for NDI frames.

Provides lightweight analysis suitable for live display:
- resolution
- mean brightness
- per-channel histograms (reduced)
- simple motion score (mean absolute difference from previous frame)
"""
from __future__ import annotations
from typing import Optional, Dict
try:
    import numpy as np
except Exception:
    np = None  # type: ignore


def analyze_frame(frame: object, prev_frame: Optional[object] = None) -> Dict:
    """Analyze a single frame (NumPy array HxWxC, BGR or Gray).

    Returns a dict with keys: width, height, mean_brightness, hist (dict), motion_score
    """
    out = {}
    if frame is None:
        return out
    if np is None:
        return out

    arr = None
    try:
        if isinstance(frame, np.ndarray):
            arr = frame
        else:
            arr = np.asarray(frame)
    except Exception:
        return out

    if arr is None:
        return out

    h = int(arr.shape[0])
    w = int(arr.shape[1]) if arr.ndim >= 2 else 0
    out['width'] = w
    out['height'] = h

    # Convert to grayscale for many metrics
    gray = None
    try:
        if arr.ndim == 3 and arr.shape[2] >= 3:
            # Assume BGR ordering
            gray = (0.114 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.299 * arr[:, :, 2]).astype('float32')
        else:
            gray = arr.astype('float32')
    except Exception:
        gray = None

    if gray is not None:
        out['mean_brightness'] = float(gray.mean())
        # coarse histogram: 16 bins
        try:
            hist, _ = np.histogram(gray.flatten(), bins=16, range=(0, 255))
            out['histogram'] = hist.tolist()
        except Exception:
            out['histogram'] = []
    else:
        out['mean_brightness'] = None
        out['histogram'] = []

    # Per-channel mean
    if arr.ndim == 3 and arr.shape[2] >= 3:
        try:
            out['mean_b'] = float(arr[:, :, 0].mean())
            out['mean_g'] = float(arr[:, :, 1].mean())
            out['mean_r'] = float(arr[:, :, 2].mean())
        except Exception:
            pass

    # Motion score: mean absolute diff from previous frame (grayscale)
    if prev_frame is not None and isinstance(prev_frame, np.ndarray) and gray is not None:
        try:
            prev = np.asarray(prev_frame)
            if prev.ndim == 3 and prev.shape[2] >= 3:
                prev_gray = (0.114 * prev[:, :, 0] + 0.587 * prev[:, :, 1] + 0.299 * prev[:, :, 2]).astype('float32')
            else:
                prev_gray = prev.astype('float32')
            diff = np.abs(gray - prev_gray)
            out['motion_score'] = float(diff.mean())
        except Exception:
            out['motion_score'] = None
    else:
        out['motion_score'] = None

    return out
