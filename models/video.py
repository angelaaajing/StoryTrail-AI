import subprocess
import json
from pathlib import Path
import os
import ffmpeg


def ffprobe(video_path: str):
    """Use ffprobe to extract video infomation"""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe error: {result.stderr}")
    return json.loads(result.stdout)


def extract_frames(video_path: str, out_dir: str, every_n_seconds: int = 2,
                   max_frames: int = 50):
    """
    Extract frames from video every `every_n_seconds` seconds.
    Saves frames as JPG in out_dir and returns list of frame file paths.
    Requires ffmpeg and ffprobe installed on system.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    probe = ffprobe(video_path)
    duration = float(probe['format']['duration'])
    frames = []
    t = 0.0
    count = 0
    while t < duration and count < max_frames:
        out_file = os.path.join(out_dir, f"frame_{count:04d}.jpg")
        (
            ffmpeg
            .input(video_path, ss=t)
            .filter('scale', 640, -1)
            .output(out_file, vframes=1)
            .overwrite_output()
            .run(quiet=True)
        )
        if os.path.exists(out_file):
            frames.append(out_file)
        t += every_n_seconds
        count += 1
    return frames
