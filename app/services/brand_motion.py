"""Reel footage that MOVES, generated locally — the motion twin of brand_footage.

`brand_footage` fixed *what appears* in a Reel: theological search terms have no
honest stock photography, so every frame is generated from a fixed subject
vocabulary under an SDXL negative prompt that excludes people, faces, figures
and religious iconography. That made an off-brand frame structurally impossible
rather than merely unlikely.

This module fixes *that the frame is a photograph*. A still under a slow push
reads as a slideshow next to reference Reels that actually move, so each scene
here is a real ~5s video clip.

**Why image-to-video, and not text-to-video.** A text-to-video model has its own
prompt adherence and its own negative prompt, so handing it "the end times"
reopens exactly the hole brand_footage was written to close — the guarantee
would move from SDXL's negative prompt, which we have tuned and trust, to a
video model's, which we have not. Instead every clip is seeded from a still that
brand_footage already generated and already vetted. The video model never sees a
free-text subject; it only animates a frame that was safe before it arrived. The
brand risk is therefore unchanged from today, and the clips look like the
account's existing imagery because they *are* the account's existing imagery,
moving.

**Why a pool, and not inline generation.** A clip costs minutes of GPU, against
a still's ~15 seconds. Generating inline would block the content scheduler on
the GPU and put a multi-minute diffusion run inside a publish path that is
supposed to be quick and interruptible. So clips are produced ahead of time by
`top_up()` on an overnight timer, and a render only ever *reads* the pool. If
the pool is empty the caller falls back to stills, which is a slightly worse
Reel rather than a failed one.

**Why the pool needs VRAM arbitration.** The 3090 is shared. Ollama pins ~12.7GB
for 30 minutes after any request — including requests that land inside the
overnight window — and ComfyUI holds ~4.9GB resident. A video job needs most of
the card, so `_ensure_vram()` measures first, evicts Ollama's resident model
when it is in the way, and retries. It never assumes the card is free.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import time
import urllib.parse
from typing import Iterable, List, Optional

import requests
from loguru import logger

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect  # noqa: F401  (typing parity)
from app.services import brand_footage, verse_card
from app.utils import utils

# --- geometry ----------------------------------------------------------------

# 9:16 at a size both engines handle natively. Wan 2.2 TI2V-5B is trained at
# 1280x704 and its high-compression VAE degrades off-bucket; LTXV wants both
# dimensions divisible by 32. 704x1280 satisfies each, and is the portrait
# orientation the Reel needs, so nothing is cropped later.
WIDTH, HEIGHT = 704, 1280

# Per-engine frame budget. Both land on ~5s, but at each model's native rate:
# Wan 2.2 is a 24fps model and wants length = 4n+1; LTXV 0.9.8 is a 30fps model
# and wants 8n+1. Asking either for the other's cadence produces stutter.
ENGINES: dict[str, dict] = {
    "wan": {
        "unet": "wan2.2_ti2v_5B_fp16.safetensors",
        "clip": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "clip_type": "wan",
        "vae": "wan2.2_vae.safetensors",
        "frames": 121,          # 4n+1 -> 5.04s
        "fps": 24.0,
        "steps": 30,
        "cfg": 5.0,
        "sampler": "uni_pc",
        "scheduler": "simple",
        "shift": 8.0,
        "vram_mib": 17000,
    },
    # LTXV lives in models/checkpoints (VAE is inside the checkpoint), and its
    # sigmas come from LTXVScheduler — see _ltxv_workflow. fp8 weights on an
    # Ampere card save VRAM but buy no compute: the 3090 has no FP8 tensor
    # cores, so ComfyUI upcasts to bf16 to do the maths.
    "ltxv": {
        "unet": "ltxv-13b-0.9.8-distilled-fp8.safetensors",
        "clip": "t5xxl_fp16.safetensors",
        "clip_type": "ltxv",
        "vae": None,
        "frames": 153,          # 8n+1 -> 5.10s
        "fps": 30.0,
        "steps": 8,             # distilled: 8 steps, and CFG disabled (1.0)
        "cfg": 1.0,
        "sampler": "euler",
        "max_shift": 2.05,
        "base_shift": 0.95,
        "terminal": 0.1,
        "vram_mib": 17000,
    },
    "ltxv2b": {
        "unet": "ltxv-2b-0.9.8-distilled.safetensors",
        "clip": "t5xxl_fp16.safetensors",
        "clip_type": "ltxv",
        "vae": None,
        "frames": 153,
        "fps": 30.0,
        "steps": 8,
        "cfg": 1.0,
        "sampler": "euler",
        "max_shift": 2.05,
        "base_shift": 0.95,
        "terminal": 0.1,
        "vram_mib": 11000,
    },
}

# What the clip should DO.
#
# The first version of this asked for "subtle natural motion, barely-moving
# camera", and it produced FROZEN clips: measured per-frame deltas of 0.48 and
# 0.97 for the pastel hills and the sunlit fields, against 5.17 for ocean surf.
# Two lessons, both counter-intuitive:
#
# 1. Most of the subject vocabulary is inherently STILL. Hills do not move.
#    Asked to animate a photograph of hills, Wan correctly animates almost
#    nothing — the failure is in the request, not the model. A clip only moves if
#    the prompt names something that plausibly moves in that scene.
# 2. Suppressing camera movement removes the last guarantee of motion. A slow
#    deliberate push is what makes a static landscape feel alive, and it is what
#    the reference Reels actually do. The thing to suppress is *fast or shaky*
#    camera, not camera movement itself.
#
# So the prompt now always asks for a slow camera move, and adds subject-specific
# element motion from MOTION_BY_KEYWORD below.
MOTION_HINT = (
    "slow steady cinematic camera push in, gentle parallax, "
    "smooth continuous movement throughout, unhurried, "
    "photographic, soft natural light, high detail, shallow depth of field"
)

# What plausibly moves, by what is in the scene. Matched against the subject
# text, most specific first, so "rain on a window" gets streaking water rather
# than the generic cloud drift. Every entry describes motion that is *already
# true* of the subject — inventing motion a scene cannot have is what produces
# warping.
MOTION_BY_KEYWORD: tuple[tuple[tuple[str, ...], str], ...] = (
    (("rain", "rain-streaked"), "raindrops running down the glass, water beading and streaking"),
    (("snowfall", "snow", "first snow"), "snowflakes drifting down slowly through still air"),
    (("wave", "ocean", "sea", "tide", "shore", "harbour"),
     "waves rolling in and receding, foam sliding over wet sand"),
    (("stream", "river", "brook"), "water flowing over stones, gentle ripples moving downstream"),
    (("lake", "still water", "reflection"),
     "water surface rippling faintly, reflections shimmering"),
    (("mist", "fog", "haze"), "mist drifting slowly across the frame, thinning and shifting"),
    (("wheat", "grass", "wildflower", "meadow", "hedgerow"),
     "stalks bending and swaying in a light breeze"),
    (("dune", "desert", "sand"), "fine sand streaming over the crest in the wind"),
    (("pine", "forest", "tree", "birch", "olive", "evergreen", "leaves", "branch"),
     "leaves and branches stirring gently, dappled light moving"),
    (("lamplight", "window", "candle"), "light flickering softly, faint movement beyond the glass"),
    # LAST, and deliberately so: "sky" and "cloud" appear in most of the
    # vocabulary ("desert dunes under a soft dawn sky"), so matching them early
    # steals subjects that have something better to move. This is the catch-all
    # for scenes whose only moving element really is the sky.
    (("cloud", "sky", "overcast"), "clouds drifting slowly across the sky, light shifting beneath"),
)

# For subjects with nothing that plausibly moves — hills, a wooden bench, a
# country path — the camera has to carry the shot alone. Atmospheric haze and
# shifting light are the only honest additions to a static scene.
_MOTION_DEFAULT = (
    "air shimmering faintly, light slowly shifting across the landscape, "
    "distant haze drifting"
)


def motion_for(subject: str) -> str:
    """What should visibly move in this subject's clip."""
    text = (subject or "").lower()
    for keywords, motion in MOTION_BY_KEYWORD:
        if any(keyword in text for keyword in keywords):
            return motion
    return _MOTION_DEFAULT

# Wan 2.2's OWN negative prompt, verbatim from ComfyUI's reference workflow
# (comfyui_workflow_templates_media_video/templates/video_wan2_2_5B_ti2v.json).
#
# It is in Chinese because the model was trained that way, and translating it
# measurably weakens it — an English list of the same concepts is simply a
# weaker conditioning signal for this checkpoint. Keeping the original is worth
# more than making this line readable, so here is what it actually asks for:
# garish/oversaturated tones, overexposure, STATIC and MOTIONLESS frames, blurry
# detail, subtitles, overall greyness, worst/low quality, JPEG artifacts,
# malformed limbs and hands, cluttered backgrounds, and crowds in the
# background. Three of those — 静态 / 静止不动的画面 / 画面静止 — are why this
# matters here: a frozen clip is the failure mode this whole module exists to
# eliminate, and the model's own vocabulary suppresses it far better than the
# word "static" does.
_WAN_OFFICIAL_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

# Appended to the official list. The still is already clean, so this exists so
# the animator does not invent what the still does not contain: a video model
# will happily walk someone into an empty landscape given 121 frames to do it
# in. Kept in English deliberately — these are our additions, and umt5 handles
# both languages in one prompt.
_BRAND_NEGATIVE = (
    "people, person, human, figure, face, hands, crowd, animal, "
    "text, watermark, logo, camera shake, fast motion, timelapse, "
    "zoom burst, warping, morphing, flicker, distortion"
)

MOTION_NEGATIVE = f"{_WAN_OFFICIAL_NEGATIVE}，{_BRAND_NEGATIVE}"

# Two numbers, because one is not enough — established by measurement on
# 2026-08-17, after a first version of this gate got it wrong.
#
# `delta` (mean change between consecutive frames) was the original single
# signal, and it fails on smooth imagery: a slow push across a flat green field
# moves very few pixels, so a clip that IS moving scored 0.76 and looked
# "frozen", while breaking surf scored 5.17 because foam has texture. Per-frame
# delta measures TEXTURE as much as motion.
#
# `drift` (first frame vs last) is what actually answers "did this scene change
# over five seconds", independent of how detailed it is. Measured:
#
#   subject                     drift   delta   looks like
#   ocean surf (Wan)            17.06    5.17   clearly moving, publishable
#   open fields (Wan)            4.82    1.31   near-still on a phone
#   pastel hills (Wan)           2.90    0.76   static
#   LTXV 13B, broken sampler     3.72    0.40   frozen
#   LTXV 2B, sand                17.75    9.56   moving, but shimmery
#
# So drift is the floor (is it alive) and delta is the ceiling (is it calm).
# MOTION_MIN_DRIFT sits at 8.0: above the 4.82 that reads as a near-still, well
# below the 17 that clearly works.
MOTION_MIN_DRIFT = 8.0
MOTION_MAX_DELTA = 8.5


# --- config helpers ----------------------------------------------------------


def _cfg(key: str, default):
    value = config.app.get(key, default)
    return default if value in (None, "") else value


def _cfg_int(key: str, default: int, *, minimum: int = 0) -> int:
    try:
        return max(minimum, int(_cfg(key, default)))
    except (TypeError, ValueError):
        return default


def _cfg_bool(key: str, default: bool) -> bool:
    value = config.app.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _comfy_url() -> str:
    """The same ComfyUI the stills come from — one client, one endpoint."""
    return verse_card._comfy_url()


def _ollama_url() -> str:
    # Ollama binds 127.0.0.1 on the host, so the LAN address does not work from
    # inside a container. Both containers sit on `ai-backend`, so use its name.
    return str(_cfg("ollama_base_url", "http://ollama:11434")).rstrip("/")


def engine_name() -> str:
    name = str(_cfg("motion_engine", "wan")).strip().lower()
    if name not in ENGINES:
        logger.warning(f"unknown motion_engine {name!r}; falling back to wan")
        return "wan"
    return name


def pool_dir() -> str:
    return utils.storage_dir("motion_pool", create=True)


def pool_target() -> int:
    """How many clips the pool should hold, capped at what is actually reachable.

    Uncapped, a target above the number of (subject, variant) slots is a target
    the pool can never meet: `missing` would never reach zero, every run would
    re-walk a full slot list to generate nothing, and `--measure` would exit
    non-zero forever. 23 animatable subjects x 4 variants = 92 slots, so the
    90-clip default fits — but the cap keeps that true if the vocabulary or the
    moving-subject filter changes.
    """
    configured = _cfg_int("motion_pool_target_clips", 90, minimum=0)
    return min(configured, len(moving_subjects()) * variants_per_subject())


def variants_per_subject() -> int:
    """Distinct clips per subject.

    Four rather than three: the moving-subject filter cut the vocabulary from 30
    to 23, and 23x3=69 clips is only 23 Reels at three clips each. 23x4=92 keeps
    the pool a month deep, which is what it is for.
    """
    return _cfg_int("motion_pool_variants_per_subject", 4, minimum=1)


# --- VRAM arbitration --------------------------------------------------------


def _gpu_free_mib() -> Optional[int]:
    """Free VRAM on the whole card, or None if it cannot be measured.

    Prefers nvidia-smi (present in this container and authoritative for the
    device) and falls back to ComfyUI's own report, which is what we actually
    care about if the two ever disagree.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode == 0 and out.stdout.strip():
            return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    try:
        stats = requests.get(f"{_comfy_url()}/system_stats", timeout=10).json()
        devices = stats.get("devices") or []
        if devices:
            return int(devices[0].get("vram_free", 0)) // (1024 * 1024)
    except (requests.exceptions.RequestException, ValueError, KeyError, IndexError):
        pass
    return None


def _loaded_ollama_models() -> list[dict]:
    try:
        resp = requests.get(f"{_ollama_url()}/api/ps", timeout=10)
        resp.raise_for_status()
        return list(resp.json().get("models") or [])
    except (requests.exceptions.RequestException, ValueError):
        return []


def evict_ollama() -> int:
    """Unload every resident Ollama model. Returns MiB we expect to reclaim.

    Ollama holds a model for OLLAMA_KEEP_ALIVE (30m here) after the last
    request, and Open WebUI's health checks are frequent enough that a stray
    call inside the overnight window re-pins ~12.7GB. A generate call with
    keep_alive=0 drops it immediately; the next real request reloads it, which
    costs that request a few seconds and nothing else.
    """
    reclaimed = 0
    for model in _loaded_ollama_models():
        name = model.get("name") or model.get("model")
        if not name:
            continue
        size = int(model.get("size_vram", 0)) // (1024 * 1024)
        try:
            requests.post(
                f"{_ollama_url()}/api/generate",
                json={"model": name, "keep_alive": 0},
                timeout=60,
            ).raise_for_status()
            logger.info(f"evicted ollama model {name} (~{size} MiB)")
            reclaimed += size
        except requests.exceptions.RequestException as exc:
            logger.warning(f"could not evict ollama model {name}: {exc}")
    return reclaimed


def unload_comfyui_models() -> bool:
    """Ask ComfyUI to drop its resident checkpoint and cached models.

    ComfyUI keeps the last SDXL checkpoint resident (~4.9GB here). It *will*
    evict that on its own when a workflow needs the room, but only once the
    workflow is already executing — which is too late to be visible to a
    pre-flight VRAM check, and makes an otherwise-fine job look impossible.
    Asking explicitly turns that latent headroom into measured headroom.
    """
    try:
        requests.post(
            f"{_comfy_url()}/free",
            json={"unload_models": True, "free_memory": True},
            timeout=60,
        ).raise_for_status()
        return True
    except requests.exceptions.RequestException as exc:
        logger.warning(f"could not ask ComfyUI to free VRAM: {exc}")
        return False


class _VramContended(RuntimeError):
    """The card was too full to start a video job, and we could not free it.

    Distinct from a generation failure on purpose. Contention is a statement
    about the *machine at this instant*, not about the clip: on 2026-08-18 two
    slots were abandoned this way and the very next slot started 17 seconds
    later with 18GB free, because the eviction had in fact worked and Ollama had
    simply reloaded mid-measurement. Counting that as a failure paged the
    operator for a run that produced 19 good clips and stopped cleanly.
    """


def _ensure_vram(need_mib: int, *, attempts: int = 3) -> bool:
    """Make room for a video job, or report that we could not.

    Never assumes the card is free. Measures first, then reclaims in increasing
    order of disruption — ComfyUI's own cached checkpoint costs nothing to drop,
    while evicting Ollama makes the next chat request reload its model — and
    measures again after each round rather than trusting that a reclaim worked.

    Some of the card is simply not ours: the reranker's ~2.5GB is a resident
    systemd service and sunshine's ~260MB is a display stream. Neither is
    evictable from here, so they are part of the budget, not an obstacle to it.
    """
    for attempt in range(1, attempts + 1):
        free = _gpu_free_mib()
        if free is None:
            logger.warning("cannot measure VRAM; proceeding and letting ComfyUI decide")
            return True
        if free >= need_mib:
            logger.info(f"VRAM ok: {free} MiB free >= {need_mib} MiB needed")
            return True
        logger.info(
            f"VRAM pressure (attempt {attempt}/{attempts}): "
            f"{free} MiB free, need {need_mib} MiB"
        )
        reclaimed_any = unload_comfyui_models()
        if evict_ollama():
            reclaimed_any = True
        if not reclaimed_any:
            # Nothing left that we are permitted to evict; further rounds would
            # measure the same number and waste the window.
            break
        time.sleep(8)   # let the driver actually return the pages

    free = _gpu_free_mib()
    logger.error(f"insufficient VRAM for a video job: {free} MiB free, need {need_mib} MiB")
    return False


# --- ComfyUI client (shared shape with verse_card.generate_background) --------


def _upload_start_image(path: str) -> Optional[str]:
    """Put the seed still on the ComfyUI server and return its input filename.

    ComfyUI's LoadImage reads from its own input directory, and ComfyUI runs in
    a different container from this code, so a local path means nothing to it.
    """
    ext = os.path.splitext(path)[1].lower() or ".png"
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    name = f"motion-seed-{utils.md5(path + str(os.path.getmtime(path)))}{ext}"
    try:
        with open(path, "rb") as fh:
            resp = requests.post(
                f"{_comfy_url()}/upload/image",
                files={"image": (name, fh, mime)},
                data={"type": "input", "overwrite": "true"},
                timeout=120,
            )
        resp.raise_for_status()
        body = resp.json()
    except (requests.exceptions.RequestException, ValueError) as exc:
        logger.error(f"could not upload seed frame to ComfyUI: {exc}")
        return None
    uploaded = body.get("name") or name
    sub = body.get("subfolder") or ""
    return f"{sub}/{uploaded}" if sub else uploaded


def _wan_workflow(spec: dict, image_name: str, prompt: str, seed: int) -> dict:
    """Wan 2.2 TI2V-5B, image-to-video, via ComfyUI's native nodes."""
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": spec["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": spec["clip"], "type": spec["clip_type"]}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": spec["vae"]}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": MOTION_NEGATIVE, "clip": ["2", 0]}},
        "7": {"class_type": "Wan22ImageToVideoLatent",
              "inputs": {"vae": ["3", 0], "width": WIDTH, "height": HEIGHT,
                         "length": spec["frames"], "batch_size": 1,
                         "start_image": ["4", 0]}},
        "8": {"class_type": "ModelSamplingSD3",
              "inputs": {"model": ["1", 0], "shift": spec["shift"]}},
        "9": {"class_type": "KSampler",
              "inputs": {"model": ["8", 0], "seed": seed, "steps": spec["steps"],
                         "cfg": spec["cfg"], "sampler_name": spec["sampler"],
                         "scheduler": spec["scheduler"], "denoise": 1.0,
                         "positive": ["5", 0], "negative": ["6", 0],
                         "latent_image": ["7", 0]}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["3", 0]}},
        "11": {"class_type": "CreateVideo",
               "inputs": {"images": ["10", 0], "fps": spec["fps"]}},
        "12": {"class_type": "SaveVideo",
               "inputs": {"video": ["11", 0], "filename_prefix": "brandmotion",
                          "format": "mp4", "codec": "h264"}},
    }


def _ltxv_workflow(spec: dict, image_name: str, prompt: str, seed: int) -> dict:
    """LTX-Video distilled, image-to-video, via ComfyUI's native nodes.

    Two structural differences from the Wan graph, both load-bearing:

    - LTXV ships its VAE inside the checkpoint, so it loads through
      CheckpointLoaderSimple (from `models/checkpoints`, NOT
      `models/diffusion_models` — putting it in the latter fails validation with
      "ckpt_name not in list").
    - The sigmas come from LTXVScheduler through SamplerCustom, not from a
      KSampler scheduler name. This is not a refinement: wired to a plain
      KSampler at 8 steps the distilled model under-denoises and returns an
      essentially FROZEN clip — measured at a frame-to-frame delta of 0.40
      against Wan's 5.17 on 2026-08-17. A still is exactly what this module
      exists to stop producing, so the custom sampler is mandatory here.
    """
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": spec["unet"]}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": spec["clip"], "type": spec["clip_type"]}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": MOTION_NEGATIVE, "clip": ["2", 0]}},
        "7": {"class_type": "LTXVImgToVideo",
              "inputs": {"positive": ["5", 0], "negative": ["6", 0], "vae": ["1", 2],
                         "image": ["4", 0], "width": WIDTH, "height": HEIGHT,
                         "length": spec["frames"], "batch_size": 1, "strength": 1.0}},
        "8": {"class_type": "LTXVConditioning",
              "inputs": {"positive": ["7", 0], "negative": ["7", 1],
                         "frame_rate": spec["fps"]}},
        "13": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": spec["sampler"]}},
        "14": {"class_type": "LTXVScheduler",
               "inputs": {"steps": spec["steps"], "max_shift": spec["max_shift"],
                          "base_shift": spec["base_shift"], "stretch": True,
                          "terminal": spec["terminal"], "latent": ["7", 2]}},
        "9": {"class_type": "SamplerCustom",
              "inputs": {"model": ["1", 0], "add_noise": True, "noise_seed": seed,
                         "cfg": spec["cfg"], "positive": ["8", 0], "negative": ["8", 1],
                         "sampler": ["13", 0], "sigmas": ["14", 0],
                         "latent_image": ["7", 2]}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["1", 2]}},
        "11": {"class_type": "CreateVideo",
               "inputs": {"images": ["10", 0], "fps": spec["fps"]}},
        "12": {"class_type": "SaveVideo",
               "inputs": {"video": ["11", 0], "filename_prefix": "brandmotion",
                          "format": "mp4", "codec": "h264"}},
    }


def _build_workflow(engine: str, spec: dict, image_name: str, prompt: str, seed: int) -> dict:
    if engine == "wan":
        return _wan_workflow(spec, image_name, prompt, seed)
    return _ltxv_workflow(spec, image_name, prompt, seed)


def _outputs_of(entry: dict) -> Iterable[dict]:
    """Every saved file in a /history entry, whatever key the node used.

    Image nodes report under `images`, video nodes under `videos`, and the key
    has moved between ComfyUI versions — so scan rather than assume.
    """
    for out in (entry.get("outputs") or {}).values():
        for key in ("videos", "images", "gifs", "files"):
            for item in out.get(key, []) or []:
                if isinstance(item, dict) and item.get("filename"):
                    yield item


def _submit_and_wait(workflow: dict, timeout: int) -> Optional[dict]:
    """POST the workflow, poll /history, return the first saved output."""
    base = _comfy_url()
    resp = None
    try:
        resp = requests.post(f"{base}/prompt", json={"prompt": workflow}, timeout=30)
        resp.raise_for_status()
        prompt_id = resp.json()["prompt_id"]
    except (requests.exceptions.RequestException, KeyError, ValueError) as exc:
        # ComfyUI returns the offending node and reason in the 400 body, which is
        # the only useful signal when a workflow is malformed. Surface it.
        detail = f" — {resp.text[:400]}" if resp is not None else ""
        logger.error(f"ComfyUI submit failed: {exc}{detail}")
        return None

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            history = requests.get(f"{base}/history/{prompt_id}", timeout=15).json()
        except (requests.exceptions.RequestException, ValueError):
            time.sleep(3)
            continue
        entry = history.get(prompt_id)
        if entry:
            status = (entry.get("status") or {}).get("status_str")
            if status == "error":
                logger.error(f"ComfyUI reported an error: {json.dumps(entry.get('status'))[:500]}")
                return None
            if entry.get("outputs"):
                for item in _outputs_of(entry):
                    return item
                logger.error("ComfyUI finished but produced no video")
                return None
        time.sleep(3)
    logger.error(f"ComfyUI timed out after {timeout}s")
    return None


def _download_output(item: dict, dest: str) -> bool:
    query = urllib.parse.urlencode({
        "filename": item["filename"],
        "subfolder": item.get("subfolder", ""),
        "type": item.get("type", "output"),
    })
    try:
        raw = requests.get(f"{_comfy_url()}/view?{query}", timeout=300).content
    except requests.exceptions.RequestException as exc:
        logger.error(f"could not fetch generated clip: {exc}")
        return False
    if not raw:
        logger.error("generated clip was empty")
        return False
    tmp = f"{dest}.part"
    try:
        with open(tmp, "wb") as fh:
            fh.write(raw)
        os.replace(tmp, dest)   # atomic: a half-written clip is never poolable
    except OSError as exc:
        logger.error(f"could not write generated clip: {exc}")
        return False
    return True


# --- clip generation ---------------------------------------------------------


def measure_motion(path: str, *, samples: int = 16) -> Optional[dict]:
    """How much a clip moves: {"drift": first-vs-last, "delta": frame-to-frame}.

    These separate a clip worth publishing from one that only looks fine as a
    thumbnail. A dead clip is identical to a good one in every still frame and
    in every file-level check — duration, resolution, codec, bitrate all pass —
    so nothing but a temporal measurement catches it.

    Decodes to raw greyscale through ffmpeg rather than reading the container's
    bitrate, because bitrate conflates motion with texture: the frozen LTXV clip
    was 244KB while a busy static landscape can be large.
    """
    try:
        import numpy as np
    except ImportError:
        logger.warning("numpy unavailable; skipping motion measurement")
        return None
    w, h = WIDTH // 4, HEIGHT // 4
    cmd = [
        "ffmpeg", "-v", "error", "-i", path,
        "-vf", f"fps={samples}/5,scale={w}:{h},format=gray",
        "-f", "rawvideo", "-",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=180)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning(f"could not measure motion: {exc}")
        return None
    buf = np.frombuffer(out.stdout, dtype=np.uint8)
    count = buf.size // (w * h)
    if count < 2:
        logger.warning(f"could not decode enough frames to measure motion ({count})")
        return None
    frames = buf[: count * w * h].reshape(count, h, w).astype(np.int16)
    deltas = [float(np.abs(frames[i + 1] - frames[i]).mean()) for i in range(count - 1)]
    return {
        "drift": float(np.abs(frames[-1] - frames[0]).mean()),
        "delta": sum(deltas) / len(deltas),
    }


def _motion_verdict(motion: Optional[dict]) -> str:
    """Classify a clip: alive enough to publish, and calm enough to be on-brand."""
    if not motion:
        return "unmeasured"
    if motion["drift"] < MOTION_MIN_DRIFT:
        # Not necessarily literally frozen — "open fields" drifted 4.82 and did
        # change — but too little to read as motion on a phone.
        return "frozen"
    if motion["delta"] > MOTION_MAX_DELTA:
        return "restless"
    return "calm"


# --- the seed still, at higher fidelity than a card needs ---------------------

# Generated ABOVE the video model's input size and downscaled into it. Wan takes
# 704x1280; feeding it 1024x1792 and letting it resample down is free apparent
# sharpness, because a downscale concentrates detail instead of inventing it.
SEED_BASE = (768, 1344)      # verse_card's own "story" bucket — native SDXL
SEED_DETAIL = (1024, 1792)   # ~1.33x, the second pass


# Why the account's imagery does not look like a photograph, and it is not the
# model's fault: `verse_card.STYLE_SUFFIX` asks for "muted warm tones, minimal
# composition, negative space". Those are graphic-design instructions, and SDXL
# obeys them — producing the smooth, idealised, almost-illustrated look that
# reads as "AI" no matter how many steps it is given.
#
# Realism therefore costs brand aesthetic, which is a decision about the account
# and not one to make inside a diffusion workflow. So this is a separate style
# used for MOTION SEEDS ONLY: it keeps the palette calm and the composition
# serene, but asks for the things that actually signal a photograph — a real
# film stock, foreground texture, atmospheric depth, optical imperfection. The
# verse cards and carousels keep `verse_card.STYLE_SUFFIX` untouched.
#
# Override with `motion_seed_style` to taste, or set it to verse_card's own
# suffix to make Reels match the cards exactly again.
SEED_STYLE = (
    "photorealistic, 35mm film photograph, Kodak Portra 400, natural daylight, "
    "fine foreground texture, atmospheric depth and haze, shallow depth of field, "
    "subtle lens vignetting, delicate film grain, serene, calm, unpeopled, "
    "high dynamic range in shadows, no harsh contrast"
)


def _seed_style() -> str:
    return str(_cfg("motion_seed_style", SEED_STYLE))


def _seed_workflow(subject: str, seed: int) -> dict:
    """Juggernaut-XL with a second low-denoise pass at higher resolution.

    Deliberately NOT a change to `verse_card.generate_background`. That function
    draws the verse cards and carousels that publish every day, and altering its
    sampler settings would change the look of the whole feed to improve the
    Reels — a blast radius nobody asked for. This is a private, higher-effort
    path for seed frames only; the cards are byte-for-byte unaffected.

    The second pass is where realism actually comes from. A single SDXL pass
    produces the smooth, slightly plastic look that reads as "AI"; re-sampling
    the upscaled latent at denoise ~0.45 puts genuine high-frequency texture
    back — grass blades, water surface, cloud edges — without changing the
    composition the first pass settled.

    Uses `verse_card.NEGATIVE_PROMPT` unchanged, because that prompt is the
    brand guarantee: people, faces, jesus, angels and crucifixes are excluded
    here exactly as they are for the cards.
    """
    ckpt = str(_cfg("comfyui_checkpoint",
                    "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"))
    positive = f"{subject}, {_seed_style()}"
    base_w, base_h = SEED_BASE
    hi_w, hi_h = SEED_DETAIL
    # cfg 5.0 rather than the cards' 6.0: on photoreal SDXL finetunes a lower
    # guidance gives more natural tonality and less crunched contrast, which is
    # what "photograph" rather than "render" looks like.
    cfg = float(_cfg("motion_seed_cfg", 5.0))
    steps = _cfg_int("motion_seed_steps", 40, minimum=10)
    detail_steps = _cfg_int("motion_seed_detail_steps", 18, minimum=6)
    denoise = float(_cfg("motion_seed_detail_denoise", 0.45))

    return {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode",
              "inputs": {"text": verse_card.NEGATIVE_PROMPT, "clip": ["4", 1]}},
        "5": {"class_type": "EmptyLatentImage",
              "inputs": {"width": base_w, "height": base_h, "batch_size": 1}},
        "3": {"class_type": "KSampler",
              "inputs": {"seed": seed, "steps": steps, "cfg": cfg,
                         "sampler_name": "dpmpp_2m", "scheduler": "karras",
                         "denoise": 1.0, "model": ["4", 0],
                         "positive": ["6", 0], "negative": ["7", 0],
                         "latent_image": ["5", 0]}},
        "10": {"class_type": "LatentUpscale",
               "inputs": {"samples": ["3", 0], "upscale_method": "bicubic",
                          "width": hi_w, "height": hi_h, "crop": "disabled"}},
        "11": {"class_type": "KSampler",
               "inputs": {"seed": seed + 1, "steps": detail_steps, "cfg": cfg,
                          "sampler_name": "dpmpp_2m", "scheduler": "karras",
                          "denoise": denoise, "model": ["4", 0],
                          "positive": ["6", 0], "negative": ["7", 0],
                          "latent_image": ["10", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"images": ["8", 0], "filename_prefix": "motionseed"}},
    }


def _seed_still(subject: str, variant: int) -> Optional[str]:
    """The SDXL frame this clip animates, generated for EXACTLY this subject.

    Not `brand_footage.generate_frame`: that takes a *search term* and re-maps
    it through the mood vocabulary, so passing it a subject that is already
    canonical returns a different one — asking for the misty meadow yields the
    ocean shore. Harmless when the caller only wants "something on-brand", but
    here the subject is also the video prompt, and a prompt that disagrees with
    its own seed frame is the one input an image-to-video model cannot reconcile.

    Cached under its own `motionseed-` prefix rather than sharing
    `brand_footage`'s `brand-*.jpg` key. Sharing was the original design and it
    was wrong once these stills diverged: they are drawn larger (1024x1792), at
    higher effort, and in a more photographic style than a card wants, so a
    shared key would let a card-quality still be used as a seed — or publish a
    seed as a card.
    """
    # PNG, because ComfyUI's SaveImage emits PNG and this file is an
    # intermediate: it is re-encoded by the video model, so there is no reason to
    # put a generation of JPEG loss in front of that.
    path = os.path.join(utils.storage_dir("cache_images", create=True),
                        f"motionseed-{utils.md5(f'{subject}|{variant}')}.png")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        logger.info(f"reusing seed still for {subject!r} v{variant}")
        return path

    if _cfg_bool("motion_seed_high_detail", True):
        seed = random.randint(1, 2**31 - 2)
        item = _submit_and_wait(_seed_workflow(subject, seed), timeout=900)
        if item and _download_output(item, path):
            logger.info(f"drew high-detail seed still for {subject!r} v{variant}")
            return path
        logger.warning(f"high-detail seed failed for {subject!r}; using the card path")

    # Fallback: the same generator the verse cards use. Lower effort, but the
    # same style suffix and negative prompt, so it is never off-brand.
    image = verse_card.generate_background(kind="story", subject=subject)
    if image is None:
        logger.error(f"no seed frame available for {subject!r}")
        return None
    try:
        image.convert("RGB").save(path, "PNG", optimize=True)
    except OSError as exc:
        logger.error(f"could not write seed frame: {exc}")
        return None
    return path


def clip_path(subject: str, variant: int) -> str:
    """Where this (subject, variant) clip lives.

    Keyed by subject exactly as brand_footage keys its stills, so the same
    collapsing of many search terms onto one subject happens here too — and
    `variant` is what stops a Reel from cutting between three copies of one
    shot when its three terms land on the same subject.
    """
    return os.path.join(pool_dir(), f"motion-{utils.md5(f'{subject}|{variant}')}.mp4")


def generate_clip(subject: str, variant: int = 0, *, engine: str = "",
                  timeout: int = 1800) -> Optional[str]:
    """Generate one ~5s clip for a subject, or return None.

    Cached on disk: an existing clip is reused rather than regenerated, which is
    what makes the pool a pool. Callers that want a *fresh* clip should pick an
    unused variant index, not delete this one.
    """
    dest = clip_path(subject, variant)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        logger.info(f"reusing pooled clip for {subject!r} v{variant}")
        return dest

    engine = engine or engine_name()
    spec = ENGINES[engine]

    if not _ensure_vram(spec["vram_mib"]):
        raise _VramContended(
            f"{spec['vram_mib']} MiB needed for {engine}; card too full"
        )

    still = _seed_still(subject, variant)
    if not still:
        return None

    image_name = _upload_start_image(still)
    if not image_name:
        return None

    prompt = f"{subject}, {motion_for(subject)}, {MOTION_HINT}"
    attempts = _cfg_int("motion_quality_attempts", 2, minimum=1)
    best: tuple[float, str] | None = None   # (distance from the calm band, path)

    for attempt in range(1, attempts + 1):
        seed = random.randint(1, 2**31 - 1)
        started = time.time()
        logger.info(f"generating motion clip ({engine} {WIDTH}x{HEIGHT} "
                    f"{spec['frames']}f@{spec['fps']:g}, attempt {attempt}/{attempts}): "
                    f"{subject!r} v{variant}")

        candidate = dest if attempt == 1 else f"{dest}.try{attempt}"
        item = _submit_and_wait(_build_workflow(engine, spec, image_name, prompt, seed),
                                timeout)
        if not item or not _download_output(item, candidate):
            continue

        elapsed = (time.time() - started) / 60
        motion = measure_motion(candidate)
        verdict = _motion_verdict(motion)
        shown = ("n/a" if not motion
                 else f"drift {motion['drift']:.2f} / delta {motion['delta']:.2f}")
        logger.info(f"clip in {elapsed:.1f} min, {shown} -> {verdict}: {subject!r}")

        if verdict in ("calm", "unmeasured"):
            if candidate != dest:
                os.replace(candidate, dest)
            return dest

        # Out of band. Keep it only as a fallback, and prefer whichever attempt
        # sits closest to the band — a slightly restless clip still beats none.
        distance = (MOTION_MIN_DRIFT - motion["drift"] if verdict == "frozen"
                    else motion["delta"] - MOTION_MAX_DELTA)
        if best is None or distance < best[0]:
            if best is not None and best[1] != dest:
                _discard(best[1])
            best = (distance, candidate)
        else:
            _discard(candidate)
        logger.info(f"re-rolling {subject!r}: {verdict} clips do not get published")

    if best is None:
        return None
    if best[1] != dest:
        os.replace(best[1], dest)
    logger.warning(
        f"no clip landed in the calm band for {subject!r} after {attempts} "
        f"attempt(s); keeping the closest one"
    )
    return dest


def _discard(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


# --- the pool ----------------------------------------------------------------


# Subjects that contain something which MOVES on its own. Measured, not guessed:
# Wan animates water, mist, falling snow and wind-blown vegetation convincingly
# (ocean surf drifted 17.06), and animates a static landscape barely at all
# (pastel hills 2.90, open fields 4.82 — both read as near-stills on a phone).
#
# The model is not at fault and no prompt fixes it: asked to animate a
# photograph of hills, it correctly animates almost nothing. Adding a Chinese
# negative prompt, per-subject motion cues and an explicit camera push moved
# those numbers by nothing at all.
#
# So the pool stops asking. These keywords select the ~two thirds of the
# vocabulary the model can bring to life, and the rest stay stills — which is
# not a downgrade, because a still under a slow push is exactly what those
# subjects were before this module existed.
_MOVING_KEYWORDS = (
    "wave", "ocean", "sea", "tide", "shore", "harbour",   # water in motion
    "stream", "river", "brook", "lake", "water",
    "mist", "fog", "haze",                                # air made visible
    "snow", "snowfall", "rain", "frost",                  # weather
    "wheat", "grass", "wildflower", "meadow", "hedgerow",  # wind-blown
    "cloud",                                              # only if named
)


def _subject_vocabulary(*, moving_only: bool = False) -> list[str]:
    """Every subject a clip may depict — the same allowlist the stills use.

    `moving_only` narrows to subjects containing something that moves by itself,
    which is what the pool fills. See _MOVING_KEYWORDS for why.
    """
    subjects = list(verse_card.BACKGROUND_SUBJECTS)
    for mood in brand_footage.MOODS.values():
        for subject in mood["subjects"]:
            if subject not in subjects:
                subjects.append(subject)
    if not moving_only:
        return subjects
    return [
        subject for subject in subjects
        if any(keyword in subject.lower() for keyword in _MOVING_KEYWORDS)
    ]


def moving_subjects() -> list[str]:
    """The subjects the pool will generate clips for."""
    if not _cfg_bool("motion_moving_subjects_only", True):
        return _subject_vocabulary()
    return _subject_vocabulary(moving_only=True)


def pool_clips() -> list[str]:
    """Every complete clip currently in the pool."""
    directory = pool_dir()
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return []
    return [
        os.path.join(directory, name)
        for name in names
        if name.startswith("motion-") and name.endswith(".mp4")
        and os.path.getsize(os.path.join(directory, name)) > 0
    ]


def pool_status() -> dict:
    clips = pool_clips()
    target = pool_target()
    subjects = moving_subjects()
    return {
        "clips": len(clips),
        "target": target,
        "missing": max(0, target - len(clips)),
        # How many Reels the pool can currently dress at three clips each — the
        # number that actually answers "am I covered".
        "reels_covered": len(clips) // 3,
        "moving_subjects": len(subjects),
        "slots": len(subjects) * variants_per_subject(),
        "dir": pool_dir(),
        "engine": engine_name(),
    }


def _planned_slots() -> list[tuple[str, int]]:
    """(subject, variant) pairs the pool wants, in fill order.

    Variant-major so the pool broadens before it deepens: after the first pass
    every subject has one clip, which is what a Reel drawing three *different*
    subjects actually needs. Depth only helps once breadth exists.
    """
    subjects = moving_subjects()
    slots: list[tuple[str, int]] = []
    for variant in range(variants_per_subject()):
        for subject in subjects:
            slots.append((subject, variant))
    return slots


class _PoolBusy(RuntimeError):
    """Another generator already holds the GPU for this pool."""


def _pool_lock(timeout: float = 0.0):
    """Exclusive lock around pool generation, or raise _PoolBusy.

    Learned the hard way on 2026-08-17: the hourly timer fired while a manual
    `--one`/top-up run was mid-clip, and the two runs competed for the card.
    Free VRAM went *down* as each tried to reclaim, and the third clip failed
    the pre-flight check outright. systemd will not double-start the service,
    but nothing stopped a timer run from overlapping a hand-run one.

    A lockfile is enough because both contenders are on this host and the loser
    should simply not run: the pool is idempotent and the next firing is an hour
    away.
    """
    import fcntl

    path = os.path.join(pool_dir(), ".generating.lock")
    handle = open(path, "w")
    deadline = time.time() + timeout
    while True:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            handle.write(f"{os.getpid()}\n")
            handle.flush()
            return handle
        except OSError:
            if time.time() >= deadline:
                handle.close()
                raise _PoolBusy(
                    "another motion-pool run holds the GPU; skipping this one"
                )
            time.sleep(2)


def _prepare_seeds(slots: list[tuple[str, int]], *, target: int,
                   deadline: Optional[float] = None) -> int:
    """Generate every missing seed still FIRST, in one SDXL batch.

    Purely an optimisation, but a large one. Interleaving costs a ComfyUI model
    swap per clip: SDXL in to draw the seed, ~7GB out, Wan's ~10GB in to animate
    it, and back again for the next subject. Measured, that turned a 9.1-minute
    clip into about 13. Drawing all the stills while SDXL is already resident,
    then animating them all while Wan is, pays the swap twice per run instead of
    twice per clip.

    Failures here are not fatal: generate_clip regenerates a missing seed on
    demand, so a still that could not be drawn now simply costs a swap later.
    """
    pending = [
        (subject, variant) for subject, variant in slots
        if not (os.path.exists(clip_path(subject, variant))
                and os.path.getsize(clip_path(subject, variant)) > 0)
    ]
    # Only as many as the run could actually consume.
    room = max(0, target - len(pool_clips()))
    pending = pending[:room]
    if not pending:
        return 0

    drawn = 0
    for subject, variant in pending:
        if deadline is not None and time.time() >= deadline:
            break
        if _seed_still(subject, variant):
            drawn += 1
    if drawn:
        logger.info(f"pre-drew {drawn} seed still(s) before animating")
    return drawn


def top_up(target: Optional[int] = None, *, deadline: Optional[float] = None,
           engine: str = "") -> dict:
    """Fill the pool toward `target`, stopping at `deadline` (a time.time()).

    Deliberately incremental and resumable: each clip is written atomically and
    cached by (subject, variant), so an interrupted overnight run loses at most
    the clip in flight and the next run picks up exactly where it stopped. That
    matters because the window is finite and a clip takes minutes.
    """
    target = pool_target() if target is None else max(0, target)
    engine = engine or engine_name()
    made: list[str] = []
    failed = 0
    contended = 0
    skipped_window = False

    try:
        lock = _pool_lock()
    except _PoolBusy as exc:
        logger.info(str(exc))
        status = pool_status()
        status.update({"generated": 0, "failed": 0, "contended": 0,
                       "skipped_locked": True})
        return status

    with lock:
        _prepare_seeds(_planned_slots(), target=target, deadline=deadline)
        made, failed, contended, skipped_window = _fill(target, deadline, engine)

    status = pool_status()
    status.update({"generated": len(made), "failed": failed,
                   "contended": contended, "window_closed": skipped_window})
    logger.info(f"pool top-up finished: {json.dumps(status)}")
    return status


def _fill(target: int, deadline: Optional[float], engine: str):
    """Generate missing clips until the target, the deadline, or repeated failure."""
    made: list[str] = []
    failed = 0
    contended = 0
    consecutive = 0
    skipped_window = False

    for subject, variant in _planned_slots():
        if len(pool_clips()) >= target:
            break
        if deadline is not None and time.time() >= deadline:
            logger.info("top-up window closed; stopping cleanly")
            skipped_window = True
            break
        dest = clip_path(subject, variant)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            continue

        try:
            path = generate_clip(subject, variant, engine=engine)
        except _VramContended as exc:
            # Not a failure: the machine was busy, not the clip bad. Kept out of
            # `failed` so it cannot page the operator, but still counted so a
            # night lost entirely to contention is visible in the status line.
            logger.info(f"skipping {subject!r} for now: {exc}")
            contended += 1
            consecutive += 1
            path = None
        else:
            if path:
                made.append(path)
                consecutive = 0
            else:
                failed += 1
                consecutive += 1

        # Trouble that is really about the machine repeats for every remaining
        # slot, so give up the window rather than burn it. Counted CONSECUTIVELY
        # and reset by any success: the earlier cumulative count would abandon a
        # healthy 5-hour window over three unrelated blips hours apart.
        if consecutive >= 3:
            logger.error("three consecutive clip failures; abandoning this window")
            break

    return made, failed, contended, skipped_window


# --- drawing from the pool (provider shape, mirroring brand_footage) ---------


def prune_pool(*, apply: bool = False) -> dict:
    """Move clips that fail the motion gate out of the pool.

    The gate stops a bad clip being *created*, but it cannot retroactively judge
    clips created before it existed — or under an earlier, wrong version of it.
    On 2026-08-17 the first gate scored per-frame delta only, which passes a clip
    that changes texture without going anywhere; re-measured on drift, 19 of 38
    pooled clips turned out to be near-stills.

    Rejects are moved to `motion_pool/rejected/`, not deleted: each one cost ~9
    minutes of GPU, they are useful for calibrating the thresholds, and a
    threshold that later proves too strict should not have destroyed evidence.
    """
    reject_dir = os.path.join(pool_dir(), "rejected")
    kept, rejected = [], []
    for path in pool_clips():
        verdict = _motion_verdict(measure_motion(path))
        if verdict in ("calm", "unmeasured"):
            kept.append(os.path.basename(path))
            continue
        rejected.append({"clip": os.path.basename(path), "verdict": verdict})
        if apply:
            os.makedirs(reject_dir, exist_ok=True)
            try:
                os.replace(path, os.path.join(reject_dir, os.path.basename(path)))
            except OSError as exc:
                logger.warning(f"could not move {path}: {exc}")
    return {"kept": len(kept), "rejected": len(rejected),
            "rejected_clips": rejected, "applied": apply,
            "reject_dir": reject_dir}


def clip_for_subject(subject: str, *, avoid: Optional[set] = None) -> Optional[str]:
    """A pooled clip for this subject that is not already used in this Reel."""
    taken = set(avoid or ())
    for variant in range(variants_per_subject()):
        path = clip_path(subject, variant)
        if path in taken:
            continue
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
    return None


def substitute_clip(*, avoid: Optional[set] = None) -> Optional[str]:
    """Any unused pooled clip, when the mapped subject has none.

    A partly-filled pool is the NORMAL state, not an error: it fills a few clips
    a night, the motion gate rejects some, and `prune_pool` removes others. If a
    missing subject meant falling back to stills, a Reel would abandon good
    footage because one of its three terms happened to map to a subject that is
    not pooled yet.

    Substituting is safe precisely because every subject in the pool came from
    the same allowlisted vocabulary under the same negative prompt — the worst
    case is a serene landscape that is slightly less apt, which is the tradeoff
    `brand_footage.subject_for` already makes for stills.
    """
    taken = set(avoid or ())
    for path in pool_clips():
        if path not in taken:
            return path
    return None


def asset_for_scene(term: str, beat_index: int, avoid: Optional[set] = None):
    """One pooled clip for one scene, as a MediaAsset. None if the pool cannot serve it."""
    from app.models.article import MediaAsset

    subject = brand_footage.subject_for(term, beat_index, avoid)
    path = clip_for_subject(subject)
    if not path:
        return None
    return MediaAsset(
        media_type="video",
        provider="comfyui_motion",
        url=path,
        width=WIDTH,
        height=HEIGHT,
        asset_id=os.path.basename(path),
        creator="",
        metadata_text=subject,
        license_name="Locally generated (SDXL still animated by a local video model)",
        license_url="",
        attribution_text="",
        source_page_url="",
        search_query=term,
        relevance_score=1.0,
    )


def search_videos_comfyui(search_term: str, video_aspect=None,
                          per_page: int = 1) -> List["MediaAsset"]:
    """Provider-shaped entry point: draw from the pool rather than search.

    Mirrors brand_footage.search_images_comfyui, including returning ONE item —
    but where that function generates on demand, this one only ever reads. A
    miss here means the pool has not been filled yet, and the caller should fall
    back to stills rather than wait minutes for the GPU.
    """
    from app.models.article import MediaAsset

    asset = asset_for_scene(search_term, 0)
    return [asset] if asset else []


# --- CLI ---------------------------------------------------------------------
#
# Run as `python3 -m app.services.brand_motion`, the same shape as
# content_scheduler, so the systemd timer can `docker exec` into the api
# container and inherit its storage mount, GPU and network by construction.
# That is deliberate: a second container in this compose project without the
# storage bind mount forks the publish ledger (see the 2026-08-16 split-brain
# runbook), so the pool must never grow its own container.


def _deadline_from(until: str) -> Optional[float]:
    """Turn an HH:MM in the account's CIVIL timezone into an absolute time.time().

    Civil, not the container's — and that distinction cost a whole morning of GPU
    on 2026-08-17. The host is America/Mexico_City and the container runs UTC, so
    a naive `datetime.now()` read `--until 05:30` as 05:30 **UTC** = 23:30 local.
    That had already passed, so it rolled to tomorrow and produced a 13.8-hour
    deadline instead of 4.5 — the pool kept generating until 09:17 local, hours
    after the idle window closed and straight through the working day.

    Uses the same `content_timezone` notion of local time that postiz and
    content_scheduler use to define a quota day, so "05:30" means the same
    moment everywhere in this codebase.
    """
    from datetime import datetime, timedelta

    from app.services.postiz import _local_tz

    try:
        hour, minute = (int(part) for part in until.split(":", 1))
    except (TypeError, ValueError):
        logger.warning(f"could not parse --until {until!r}; running without a deadline")
        return None
    tz = _local_tz()
    now = datetime.now(tz)
    stop = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if stop <= now:
        stop += timedelta(days=1)
    logger.info(f"top-up deadline: {stop.isoformat()} "
                f"({(stop - now).total_seconds() / 3600:.1f}h from now)")
    return stop.timestamp()


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="brand_motion",
        description="Generate and manage the pool of on-brand motion clips.",
    )
    parser.add_argument("--status", action="store_true",
                        help="print pool status as JSON and exit")
    parser.add_argument("--top-up", action="store_true",
                        help="generate clips until the pool reaches its target")
    parser.add_argument("--target", type=int, default=None,
                        help=f"override motion_pool_target_clips (default {pool_target()})")
    parser.add_argument("--until", default="",
                        help="stop starting new clips at this local HH:MM (e.g. 05:30)")
    parser.add_argument("--minutes", type=int, default=0,
                        help="stop starting new clips after this many minutes")
    parser.add_argument("--engine", default="", choices=["", *ENGINES],
                        help=f"override motion_engine (default {engine_name()})")
    parser.add_argument("--one", metavar="SUBJECT", default="",
                        help="generate a single clip for one subject, then exit")
    parser.add_argument("--variant", type=int, default=0,
                        help="variant index for --one")
    parser.add_argument("--measure", action="store_true",
                        help="report the motion of every pooled clip and exit")
    parser.add_argument("--prune", action="store_true",
                        help="move clips failing the motion gate to rejected/ (dry run)")
    parser.add_argument("--apply", action="store_true",
                        help="with --prune, actually move the files")
    args = parser.parse_args(argv)

    if args.prune:
        result = prune_pool(apply=args.apply)
        for row in result["rejected_clips"]:
            print(f"  reject {row['clip']:46s} {row['verdict']}")
        print(json.dumps({k: v for k, v in result.items()
                          if k != "rejected_clips"}, indent=2))
        if not args.apply and result["rejected"]:
            print("\n  dry run — re-run with --apply to move them")
        return 0

    if args.measure:
        rows = []
        for path in pool_clips():
            motion = measure_motion(path)
            rows.append({"clip": os.path.basename(path),
                         "drift": round(motion["drift"], 2) if motion else None,
                         "delta": round(motion["delta"], 2) if motion else None,
                         "verdict": _motion_verdict(motion)})
        print(f"  {'clip':46s} {'drift':>7s} {'delta':>7s}  verdict")
        for row in sorted(rows, key=lambda r: r["drift"] or 0):
            print(f"  {row['clip']:46s} {str(row['drift']):>7s} "
                  f"{str(row['delta']):>7s}  {row['verdict']}")
        bad = [r for r in rows if r["verdict"] not in ("calm", "unmeasured")]
        print(json.dumps({"clips": len(rows), "outside_calm_band": len(bad),
                          "min_drift": MOTION_MIN_DRIFT,
                          "max_delta": MOTION_MAX_DELTA}, indent=2))
        return 1 if bad else 0

    if args.status or not (args.top_up or args.one):
        print(json.dumps(pool_status(), indent=2))
        return 0

    if args.one:
        # Same lock as top_up, so a hand-run clip cannot collide with the timer.
        # Taken here rather than inside generate_clip: flock blocks on a second
        # descriptor in the same process, so top_up holding it would deadlock.
        try:
            lock = _pool_lock()
        except _PoolBusy as exc:
            print(json.dumps({"skipped": True, "error": str(exc)}))
            return 0
        try:
            with lock:
                path = generate_clip(args.one, args.variant, engine=args.engine)
        except _VramContended as exc:
            print(json.dumps({"subject": args.one, "variant": args.variant,
                              "skipped": True, "contended": True,
                              "error": str(exc)}))
            return 0
        print(json.dumps({"subject": args.one, "variant": args.variant, "path": path}))
        return 0 if path else 1

    deadline = None
    if args.until:
        deadline = _deadline_from(args.until)
    elif args.minutes > 0:
        deadline = time.time() + args.minutes * 60

    status = top_up(args.target, deadline=deadline, engine=args.engine)
    print(json.dumps(status, indent=2))
    # Only genuine generation failures are worth OnFailure= and an ntfy push.
    #
    # The first version returned 1 whenever the pool was still short and this run
    # made nothing, which fired an alert for two entirely normal outcomes: a run
    # that skipped because another held the lock, and a run whose deadline closed
    # before it could start a ~9-minute clip. The pool being short is its steady
    # state for the first few nights — it is not an error.
    #
    # VRAM contention joined that list on 2026-08-18, when a run that generated
    # 19 clips and stopped cleanly at its deadline still paged, because two slots
    # had bounced off a card that Ollama was reloading. It is reported as
    # `contended` in the status line and is visible in the log, but a busy GPU is
    # a fact about the hour, not a fault to wake someone for.
    return 1 if status.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
