"""Licensed music, generated ahead of time into a pool.

WHY THIS EXISTS. Every Reel the account has published carried a random track
from `resource/songs/` — 29 files with no licence file, present since the
upstream `init` commit. Undocumented-licence music on an account being grown
commercially is the same class of risk this pipeline already handles carefully
for copyrighted Bible translations and for NC/ND photographs, and Instagram
fingerprints audio aggressively: the failure mode is a Reel muted or blocked in
some territories, discovered long after the fact.

So music is now generated through the account's own ElevenLabs subscription,
where the rights are clear.

WHY A POOL RATHER THAN PER-RENDER GENERATION. `elevenlabs_music.generate_bgm`
already existed and does video-to-music: it uploads a proxy of the finished
video and scores it. That is the better *result* and the wrong *shape* here —

  - it puts a paid API call, an upload and a multi-minute wait inside the
    publish path, where a timeout costs the post;
  - it bills per render, and a Reel is often re-rendered;
  - it cannot work offline, and the rest of this pipeline deliberately can.

A pool of a dozen instrumental beds costs a dozen generations ONCE, keeps the
render path local and fast, and mirrors `brand_motion`'s clip pool — the same
pattern, for the same reasons, in the same stack.

THE FALLBACK IS SILENCE, NEVER THE BUNDLED SONGS. If the pool is empty a Reel
renders without music. Falling back to `resource/songs/` would quietly reinstate
the exact risk this module exists to remove, and a silent Reel is a content
problem while an unlicensed one is a legal one. `music_allow_bundled_songs`
exists to restore the old behaviour deliberately, and defaults to false.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import sys
import time
from typing import Optional

import requests
from loguru import logger

from app.config import config

POOL_DIRNAME = "music_pool"

# 32 seconds, not the length of a Reel. Every consumer already loops or trims
# through moviepy's AudioLoop, so one length serves a 12-second quote Reel and a
# 40-second article Reel, and the pool does not have to be rebuilt when the
# format's pacing changes.
TRACK_SECONDS = 32
MIN_TRACK_BYTES = 20 * 1024

# A DEAD TRACK PASSES EVERY STRUCTURAL CHECK.
#
# The first three generations came back as valid 32.04s stereo 192kbps mp3s of
# the right size — and one of them measured mean -52.8 dB, peak -30.8 dB. That
# is inaudible, and _music_bed then multiplies by 0.22 on top of it. Codec,
# duration, sample rate and byte count are identical between a usable bed and a
# silent one, which is the same trap the motion pool hit: a frozen clip and a
# moving clip differ in nothing a file inspection can see.
#
# So loudness is measured, and it takes two numbers for the same reason motion
# took two:
#
#   PEAK decides whether there is a performance in the file at all. A track
#   whose loudest moment is -30 dB has nothing to recover — normalising it just
#   amplifies hiss — so it is rejected and re-rolled.
#
#   LOUDNESS (integrated, EBU R128) is then normalised rather than judged. A
#   quiet-but-real bed is exactly what this account wants; it just has to arrive
#   at a predictable level, or `quote_reel_music_volume` means something
#   different for every track and the knob is unusable.
#
# Measured for reference: the bundled songs sit at mean -22.7 dB / peak -1.3 dB,
# and the two usable generations at mean -27.1/-29.7 dB, peak -4.6/-10.7 dB.
MIN_SOURCE_PEAK_DB = -20.0
TARGET_LUFS = -16.0
QUALITY_ATTEMPTS = 2

# Instrumental beds that sit UNDER on-screen text. Deliberately plain: this is a
# bed, not a track, and anything with a hook competes with the words. Overridable
# per account via the pack (`music.prompts`).
PROMPTS: list[str] = [
    "Sparse ambient instrumental. Soft sustained piano, warm analog pad, very "
    "slow. No percussion, no vocals, no melody hook. Calm, unhurried, leaves "
    "space for on-screen text.",
    "Quiet instrumental for a reflective short film. Felt piano and low strings, "
    "gentle swell, no drums, no vocals. Warm and still.",
    "Minimal ambient bed. Long sustained synth pad, faint air and tape hiss, no "
    "rhythm, no vocals. Spacious and calm.",
    "Slow acoustic guitar harmonics over a soft drone. No percussion, no vocals. "
    "Intimate, warm, unhurried.",
    "Gentle instrumental with soft cello and distant piano. Very slow, no drums, "
    "no vocals. Melancholy but hopeful.",
    "Warm ambient bed with soft bell tones and a low pad. No beat, no vocals. "
    "Peaceful, early-morning stillness.",
]

DEFAULT_TARGET = 12


def _cfg(key: str, default):
    value = config.app.get(key, default)
    return default if value in (None, "") else value


def _cfg_bool(key: str, default: bool) -> bool:
    value = config.app.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def prompts() -> list[str]:
    from app.services import pack

    return pack.typed("music.prompts", PROMPTS)


def pool_dir() -> str:
    from app.utils import utils

    return utils.storage_dir(POOL_DIRNAME, create=True)


def _track_path(prompt: str, variant: int) -> str:
    key = hashlib.md5(f"{prompt}|{variant}".encode("utf-8")).hexdigest()
    return os.path.join(pool_dir(), f"music-{key}.mp3")


def tracks() -> list[str]:
    try:
        return sorted(
            os.path.join(pool_dir(), f)
            for f in os.listdir(pool_dir())
            if f.startswith("music-") and f.endswith(".mp3")
        )
    except OSError:
        return []


def target() -> int:
    """How deep to fill. Capped at the reachable slot count.

    Same guard as the motion pool: a configured target above the number of
    (prompt, variant) slots would otherwise be a shortfall the pool can never
    clear, and a job that can never succeed eventually gets ignored.
    """
    try:
        want = int(_cfg("music_pool_target", DEFAULT_TARGET))
    except (TypeError, ValueError):
        want = DEFAULT_TARGET
    variants = max(1, int(_cfg("music_pool_variants_per_prompt", 2)))
    return max(0, min(want, len(prompts()) * variants))


# --- generation --------------------------------------------------------------


def measure(path: str) -> dict:
    """Peak and mean dBFS of an audio file, via ffmpeg's volumedetect."""
    import re
    import subprocess

    from app.utils import utils

    try:
        proc = subprocess.run(
            [utils.get_ffmpeg_binary(), "-hide_banner", "-i", path,
             "-af", "volumedetect", "-f", "null", "-"],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning(f"could not measure {os.path.basename(path)}: {exc}")
        return {}
    out = {}
    for key, field in (("mean_volume", "mean_db"), ("max_volume", "peak_db")):
        match = re.search(rf"{key}:\s*(-?[\d.]+) dB", proc.stderr)
        if match:
            out[field] = float(match.group(1))
    return out


def _normalise(path: str) -> bool:
    """Bring a track to TARGET_LUFS in place. Best effort.

    A failure here is not fatal: an un-normalised but audible bed is still a
    licensed bed, and losing the track over a loudness pass would be worse than
    it being a few dB off.
    """
    import subprocess

    from app.utils import utils

    tmp = f"{path}.norm.mp3"
    try:
        proc = subprocess.run(
            [utils.get_ffmpeg_binary(), "-hide_banner", "-y", "-i", path,
             "-af", f"loudnorm=I={TARGET_LUFS}:TP=-1.5:LRA=11",
             "-c:a", "libmp3lame", "-b:a", "192k", tmp],
            capture_output=True, text=True, timeout=300,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning(f"loudness normalisation failed for "
                       f"{os.path.basename(path)}: {exc}")
        _remove(tmp)
        return False
    if proc.returncode != 0 or not os.path.exists(tmp):
        logger.warning(f"loudness normalisation returned {proc.returncode} for "
                       f"{os.path.basename(path)}; keeping the original")
        _remove(tmp)
        return False
    try:
        os.replace(tmp, path)
    except OSError as exc:
        logger.warning(f"could not replace {path}: {exc}")
        _remove(tmp)
        return False
    return True


def generate_track(prompt: str, out_path: str, seconds: int = TRACK_SECONDS) -> bool:
    """One usable ElevenLabs track: generated, gated on peak, then normalised.

    Written atomically: an interrupted run leaves no half a track behind for a
    later render to pick up and fail on.
    """
    for attempt in range(1, QUALITY_ATTEMPTS + 1):
        if not _generate_once(prompt, out_path, seconds):
            return False
        stats = measure(out_path)
        peak = stats.get("peak_db")
        if peak is None:
            logger.warning("could not measure the generated track; keeping it")
            return True
        if peak < MIN_SOURCE_PEAK_DB:
            logger.warning(
                f"generated track is silent (peak {peak:.1f} dB, floor "
                f"{MIN_SOURCE_PEAK_DB} dB) — re-rolling "
                f"({attempt}/{QUALITY_ATTEMPTS})")
            _remove(out_path)
            continue
        before = stats.get("mean_db")
        if _normalise(out_path):
            after = measure(out_path).get("mean_db")
            logger.info(f"music: normalised {os.path.basename(out_path)} "
                        f"mean {before:.1f} -> {after:.1f} dB")
        return True
    logger.warning(f"no usable track after {QUALITY_ATTEMPTS} attempts for this prompt")
    return False


def _generate_once(prompt: str, out_path: str, seconds: int) -> bool:
    from app.services import elevenlabs_music as el

    key = el.get_api_key()
    if not key:
        logger.warning("no ElevenLabs API key; cannot generate music")
        return False

    url = f"{el._base_url()}/v1/music"
    body = {"prompt": prompt, "music_length_ms": int(seconds * 1000),
            "model_id": el._model_id()}
    tmp = f"{out_path}.part"
    try:
        response = requests.post(
            url,
            headers={"xi-api-key": key, "Content-Type": "application/json"},
            json=body,
            timeout=el._request_timeout(),
        )
    except requests.RequestException as exc:
        logger.warning(f"music generation failed: {exc}")
        return False
    if response.status_code != 200:
        logger.warning(f"music generation http {response.status_code}: "
                       f"{el._safe_response_error(response)}")
        return False
    content = response.content
    if len(content) < MIN_TRACK_BYTES:
        logger.warning(f"music generation returned {len(content)} bytes; discarding")
        return False
    try:
        with open(tmp, "wb") as fh:
            fh.write(content)
        os.replace(tmp, out_path)
    except OSError as exc:
        logger.warning(f"could not write {out_path}: {exc}")
        _remove(tmp)
        return False
    logger.info(f"music: wrote {os.path.basename(out_path)} "
                f"({len(content) // 1024} KB, {seconds}s)")
    return True


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def top_up(limit: Optional[int] = None) -> dict:
    """Fill the pool to `target()`. Idempotent; safe to run repeatedly.

    Fill order is variant-major, exactly as the motion pool fills: after the
    first pass every prompt has one track, which is what variety across
    consecutive Reels actually needs. Depth only adds more of the same mood.
    """
    want = target()
    made, failed = 0, 0
    variants = max(1, int(_cfg("music_pool_variants_per_prompt", 2)))
    for variant in range(variants):
        for prompt in prompts():
            if len(tracks()) >= want:
                return {"made": made, "failed": failed, "tracks": len(tracks()),
                        "target": want, "reason": "target met"}
            if limit is not None and made >= limit:
                return {"made": made, "failed": failed, "tracks": len(tracks()),
                        "target": want, "reason": "limit reached"}
            path = _track_path(prompt, variant)
            if os.path.exists(path):
                continue
            if generate_track(prompt, path):
                made += 1
            else:
                failed += 1
    return {"made": made, "failed": failed, "tracks": len(tracks()),
            "target": want, "reason": "swept every slot"}


# --- selection ---------------------------------------------------------------


def _recent_path() -> str:
    return os.path.join(pool_dir(), "recent.json")


def _recent() -> list[str]:
    try:
        with open(_recent_path(), encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def _remember(name: str, keep: int = 8) -> None:
    recent = [n for n in _recent() if n != name]
    recent.append(name)
    try:
        with open(_recent_path(), "w", encoding="utf-8") as fh:
            json.dump(recent[-keep:], fh)
    except OSError:
        pass


def select_track() -> str:
    """A pooled track, least-recently-used, or "" if the pool is empty.

    "" means silence. It must never mean "fall back to resource/songs".
    """
    available = tracks()
    if not available:
        return ""
    recent = _recent()
    fresh = [p for p in available if os.path.basename(p) not in recent]
    chosen = random.choice(fresh) if fresh else random.choice(available)
    _remember(os.path.basename(chosen))
    return chosen


def audit(renormalise: bool = False) -> dict:
    """Measure every pooled track; optionally bring stragglers up to target.

    Tracks generated before the loudness gate existed are quiet but real, and
    normalising them is free where regenerating them costs credit. Anything
    below the peak floor is reported as dead rather than silently amplified.
    """
    rows, fixed, dead = [], 0, []
    for path in tracks():
        stats = measure(path)
        peak, mean = stats.get("peak_db"), stats.get("mean_db")
        name = os.path.basename(path)
        if peak is not None and peak < MIN_SOURCE_PEAK_DB:
            dead.append(name)
        elif renormalise and mean is not None and mean < TARGET_LUFS - 3:
            if _normalise(path):
                fixed += 1
                stats = measure(path)
        rows.append({"track": name, **stats})
    return {"tracks": rows, "renormalised": fixed, "dead": dead}


def prune(apply: bool = False) -> dict:
    """Move tracks below the peak floor out of the pool.

    MOVED, not deleted, exactly as the motion pool does it: each one cost a paid
    generation, and a threshold that later proves too strict should not have
    destroyed the evidence.
    """
    rejected_dir = os.path.join(pool_dir(), "rejected")
    doomed = audit()["dead"]
    if apply and doomed:
        os.makedirs(rejected_dir, exist_ok=True)
        for name in doomed:
            try:
                os.replace(os.path.join(pool_dir(), name),
                           os.path.join(rejected_dir, name))
            except OSError as exc:
                logger.warning(f"could not move {name}: {exc}")
    return {"rejected": doomed, "applied": bool(apply), "dir": rejected_dir}


def status() -> dict:
    have = tracks()
    return {
        "tracks": len(have),
        "target": target(),
        "missing": max(0, target() - len(have)),
        "prompts": len(prompts()),
        "variants": int(_cfg("music_pool_variants_per_prompt", 2)),
        "dir": pool_dir(),
        "seconds_per_track": TRACK_SECONDS,
        "bundled_songs_allowed": _cfg_bool("music_allow_bundled_songs", False),
    }


# --- CLI ---------------------------------------------------------------------


def _locked(fn, *args, **kwargs):
    """One generator at a time.

    The motion pool learned this the hard way: a timer firing while someone ran
    a job by hand had both competing for the same resource. Cheaper here, but
    duplicate paid generations are worse than a skipped run.
    """
    lock_path = os.path.join(pool_dir(), ".generating.lock")
    with open(lock_path, "w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            logger.info("another music-pool job holds the lock; exiting")
            return {"skipped": "locked"}
        try:
            return fn(*args, **kwargs)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def main() -> int:
    parser = argparse.ArgumentParser(description="Licensed music pool for Reels.")
    parser.add_argument("--status", action="store_true", help="how full the pool is")
    parser.add_argument("--top-up", action="store_true", help="generate missing tracks")
    parser.add_argument("--one", metavar="PROMPT", help="generate a single track")
    parser.add_argument("--limit", type=int, help="stop after this many generations")
    parser.add_argument("--measure", action="store_true",
                        help="loudness of every pooled track")
    parser.add_argument("--renormalise", action="store_true",
                        help="with --measure, bring quiet tracks up to target")
    parser.add_argument("--prune", action="store_true",
                        help="move silent tracks to rejected/ (dry run without --apply)")
    parser.add_argument("--apply", action="store_true", help="make --prune act")
    args = parser.parse_args()

    if args.measure:
        print(json.dumps(audit(renormalise=args.renormalise), indent=2))
        return 0

    if args.prune:
        print(json.dumps(prune(apply=args.apply), indent=2))
        return 0

    if args.one:
        path = _track_path(args.one, 0)
        ok = _locked(generate_track, args.one, path)
        print(json.dumps({"ok": bool(ok), "path": path if ok else None}, indent=2))
        return 0 if ok else 1

    if args.top_up:
        result = _locked(top_up, args.limit)
        print(json.dumps(result, indent=2))
        # A short pool is its normal state while it fills, and exiting non-zero
        # for that fires OnFailure= and an ntfy push for nothing. Only a real
        # generation failure is a failure.
        return 1 if (result or {}).get("failed") else 0

    print(json.dumps(status(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
