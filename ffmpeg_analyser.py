#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from typing import List, Optional, Dict, Any


class VideoAnalyzer:
    def __init__(self, path: str, limit: Optional[int] = None):
        self.path = path
        self.limit = limit

    def _ffprobe_json(self, args: List[str]) -> Dict[str, Any]:
        cmd = ["ffprobe", "-v", "error", "-of", "json"] + args + [self.path]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, encoding="utf-8")
        return json.loads(out)

    def _ffprobe_text(self, args: List[str]) -> str:
        cmd = ["ffprobe", "-v", "error"] + args + [self.path]
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, encoding="utf-8").strip()

    def get_duration(self) -> Optional[float]:
        try:
            out = self._ffprobe_text(["-show_entries", "format=duration",
                                      "-of", "default=noprint_wrappers=1:nokey=1"])
            return float(out) if out else None
        except (subprocess.CalledProcessError, ValueError):
            return None

    def get_frames(self) -> List[Dict[str, Any]]:
        try:
            data = self._ffprobe_json([
                "-select_streams", "v",
                "-show_frames",
                "-show_entries", "frame=pkt_pts_time,pkt_dts_time,pkt_duration_time,pts_time,dts_time"
            ])
            return data.get("frames", [])
        except subprocess.CalledProcessError as e:
            print(f"ffprobe failed for {self.path}: {e}", file=sys.stderr)
            return []

    @staticmethod
    def _fmt(value: Optional[str]) -> str:
        if value is None or value == "":
            return "N/A"
        try:
            return f"{float(value):.6f}"
        except Exception:
            return str(value)

    def analyze(self) -> None:
        duration = self.get_duration()
        print(f"File: {self.path}")
        print(f"Duration (seconds): {duration if duration is not None else 'N/A'}")
        frames = self.get_frames()
        print("\nIndex\tPTS\t\tDTS\t\tDuration")
        for i, f in enumerate(frames):
            if self.limit is not None and i >= self.limit:
                break
            pts = f.get("pkt_pts_time") or f.get("pts_time")
            dts = f.get("pkt_dts_time") or f.get("dts_time")
            dur = f.get("pkt_duration_time") or ""
            print(f"{i}\t{self._fmt(pts)}\t{self._fmt(dts)}\t{self._fmt(dur)}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Print PTS, DTS and length of a video file using ffprobe.")
    p.add_argument("paths", nargs="+", help="Video file(s) to analyze")
    p.add_argument("--limit", type=int, help="Limit number of frames to print (optional)")
    args = p.parse_args(argv)
    for path in args.paths:
        analyzer = VideoAnalyzer(path, limit=args.limit)
        analyzer.analyze()


if __name__ == "__main__":