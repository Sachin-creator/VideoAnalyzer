#!/usr/bin/env python3
"""Simple SRT/URL analyser using ffprobe.

This mirrors the rtpm_analyser functionality but is provided as a
separate module for clarity. It queries ffprobe for format/stream metadata
for SRT (or other URL) inputs and prints JSON output.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from typing import List, Dict, Any, Optional


def ffprobe_json(url: str, args: List[str]) -> Optional[Dict[str, Any]]:
    cmd = ["ffprobe", "-v", "error", "-of", "json"] + args + [url]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, encoding="utf-8")
        return json.loads(out)
    except subprocess.CalledProcessError as e:
        print(f"ffprobe failed: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"failed to decode ffprobe output as JSON: {e}")
        return None


def analyze_url(url: str, show_frames: bool = False, show_packets: bool = False) -> None:
    data = ffprobe_json(url, ["-show_format", "-show_streams"]) or {}
    out = {
        "url": url,
        "format": data.get("format"),
        "streams": data.get("streams", []),
    }

    if show_frames:
        frames = ffprobe_json(url, ["-show_frames"]) or {}
        out["frames"] = frames.get("frames", [])

    if show_packets:
        packets = ffprobe_json(url, ["-show_packets"]) or {}
        out["packets"] = packets.get("packets", [])

    print(json.dumps(out, indent=2))


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="SRT/URL analyser using ffprobe")
    p.add_argument("urls", nargs="+", help="SRT/RTMP/HTTP/etc. URL(s) to analyze")
    p.add_argument("--frames", action="store_true", help="Attempt to show frame-level info (may block for live sources)")
    p.add_argument("--packets", action="store_true", help="Attempt to show packet-level info (may block for live sources)")
    args = p.parse_args(argv)

    for url in args.urls:
        try:
            analyze_url(url, show_frames=args.frames, show_packets=args.packets)
        except KeyboardInterrupt:
            print("Interrupted by user")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
