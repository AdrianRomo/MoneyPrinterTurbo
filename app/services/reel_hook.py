"""Burn a text hook into the opening seconds of a Reel.

Retention on Reels is decided in the first two seconds, and a video that opens
on narration over stock footage gives a scrolling viewer no reason to stay. A
short on-screen line is the single largest lever on Reel reach — larger than
anything about the footage itself.

Implemented as an ffmpeg pass over the finished render rather than inside the
renderer: it keeps the change additive, and if it fails the original video is
returned untouched so a hook problem can never cost a post.
"""

from __future__ import annotations

import os
import re
import subprocess

from loguru import logger

from app.config import config

FONT = "/influencer-automation-2.0/resource/fonts/BeVietnamPro-Bold.ttf"
HOOK_SECONDS = 2.6


def _cfg(key: str, default):
    value = config.app.get(key, default)
    return default if value in (None, "") else value


def generate_hook(subject: str, script: str = "") -> str:
    """A short scroll-stopping line. Falls back to the subject."""
    from app.services import llm

    prompt = (
        "Write a 4-7 word on-screen hook for the opening seconds of a short "
        "devotional video. It must create curiosity or name a feeling — not "
        "summarise. No hashtags, no quotation marks, no emoji, no full stop.\n"
        f"Topic: {subject}\n"
        f"Script opening: {script[:300]}"
    )
    try:
        raw = (llm._generate_response(prompt) or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"hook generation failed: {exc}")
        raw = ""
    line = re.sub(r'["""\'`]', "", raw.splitlines()[0] if raw else "").strip(" .")
    if not line or len(line) > 46:
        line = subject.strip()[:46]
    return line.upper()


def _fit(text: str, max_px: int, max_size: int = 78, min_size: int = 40) -> tuple[str, int]:
    """Largest size that fits, wrapping to two lines before shrinking further.

    A fixed font size clips: the first hook rendered came back as
    "ING BREW INVITES QUIET MIRA" because 35 characters at size 72 is wider
    than the frame.
    """
    from PIL import ImageDraw, ImageFont, Image

    d = ImageDraw.Draw(Image.new("RGB", (10, 10)))

    def width_at(line: str, size: int) -> float:
        return d.textlength(line, font=ImageFont.truetype(FONT, size))

    for size in range(max_size, min_size - 1, -2):
        if width_at(text, size) <= max_px:
            return text, size

    # Still too wide — split near the middle on a word boundary.
    words = text.split()
    if len(words) > 1:
        best, best_delta = 1, None
        for i in range(1, len(words)):
            delta = abs(len(" ".join(words[:i])) - len(" ".join(words[i:])))
            if best_delta is None or delta < best_delta:
                best, best_delta = i, delta
        two = " ".join(words[:best]) + "\n" + " ".join(words[best:])
        for size in range(max_size, min_size - 1, -2):
            if max(width_at(l, size) for l in two.split("\n")) <= max_px:
                return two, size
        return two, min_size
    return text, min_size


def _escape(text: str) -> str:
    # ffmpeg drawtext is fussy: colons and quotes must be escaped.
    return text.replace("\\", "").replace(":", "\\:").replace("'", "").replace("%", "")


def add_hook(video_path: str, text: str, out_path: str = "") -> str:
    """Return a copy with the hook burned in, or the original on any failure."""
    if not text or not os.path.exists(video_path):
        return video_path
    if not os.path.exists(FONT):
        logger.warning(f"hook font missing: {FONT}")
        return video_path

    out_path = out_path or video_path.replace(".mp4", "-hooked.mp4")
    fitted, font_size = _fit(text, int(1080 * 0.88))
    safe = _escape(fitted)
    # Sits above centre, clear of Instagram's bottom UI, with a stroke and a
    # short fade so it does not simply cut out.
    draw = (
        f"drawtext=fontfile={FONT}:text='{safe}':"
        f"fontcolor=white:fontsize={font_size}:borderw=6:bordercolor=black@0.85:"
        f"x=(w-text_w)/2:y=h*0.28:line_spacing=12:"
        f"enable='lt(t,{HOOK_SECONDS})':"
        f"alpha='if(lt(t,{HOOK_SECONDS - 0.4}),1,{HOOK_SECONDS}-t)'"
    )
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-i", video_path,
        "-vf", draw, "-c:a", "copy",
        "-c:v", str(_cfg("video_codec", "libx264")), "-b:v", "8000k",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=600,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        detail = getattr(exc, "stderr", b"")
        logger.warning(f"hook overlay failed, publishing the original: "
                       f"{exc.__class__.__name__} {detail[:200] if detail else ''}")
        return video_path
    if not os.path.exists(out_path) or os.path.getsize(out_path) < 1024:
        logger.warning("hook overlay produced an unusable file; using the original")
        return video_path
    logger.info(f"hook burned in: {text!r}")
    return out_path
