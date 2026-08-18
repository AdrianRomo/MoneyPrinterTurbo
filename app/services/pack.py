"""Niche packs — everything about the ACCOUNT, separated from the machinery.

The pipeline was written for one Instagram account and its domain is welded in:
scripture references, Commons nature subjects, faith hashtag sets, devotional
series, a specific wordmark. Reusing it for another account meant editing eight
modules.

A pack is one file — `packs/<name>/pack.yaml` — holding the things that are true
of an account rather than of the pipeline: brand furniture, subject vocabularies,
copy banks, hashtag sets, series, themes and cadence. `content_pack` in
config.toml names it.

THE PACK OVERRIDES; IT DOES NOT REPLACE.

Every call site passes the module's existing constant as the fallback, so a
missing pack file, a missing key, or a malformed one leaves behaviour exactly as
it is today. That matters because this lands on a pipeline that publishes to a
live account unattended: the failure mode of a typo in a pack file must be "the
default applies", never "the account posts something strange" or "the run dies".

WHAT DOES NOT GO IN A PACK.

The safety properties are pipeline behaviour, not account taste, and they stay in
code where they are reviewed:

  - the scripture/text provider contract — a model proposes a reference, a
    verified source supplies the words
  - PUBLIC_DOMAIN_TRANSLATIONS, and the refusal to publish copyrighted text
  - wikimedia's ALLOWED_LICENCES, and locations parsed from file metadata rather
    than inferred
  - the contrast gate, the aesthetic gate, the crop/upscale gates, the motion
    drift gate, the no-digits rule on unverified LLM prose
  - the quota, window, cap and review machinery

Those are the reusable part. A pack that could switch them off would make every
new account start from zero trust.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from loguru import logger

from app.config import config

DEFAULT_PACK = "holy-ordinary"
PACKS_DIR = os.path.join("/influencer-automation-2.0", "packs")

_cache: dict[str, dict] = {}


def pack_name() -> str:
    return str(config.app.get("content_pack", "") or DEFAULT_PACK).strip() or DEFAULT_PACK


def pack_path(name: Optional[str] = None) -> str:
    return os.path.join(PACKS_DIR, name or pack_name(), "pack.yaml")


def load(name: Optional[str] = None, refresh: bool = False) -> dict:
    """The pack as a dict. Never raises: an unreadable pack is an empty pack."""
    name = name or pack_name()
    if not refresh and name in _cache:
        return _cache[name]

    data: dict = {}
    path = pack_path(name)
    try:
        import yaml

        with open(path, encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
        if isinstance(loaded, dict):
            data = loaded
        else:
            logger.warning(f"pack {name!r} is not a mapping; ignoring it")
    except FileNotFoundError:
        # Entirely normal for a deployment that has not adopted packs.
        logger.debug(f"no pack at {path}; module defaults apply")
    except Exception as exc:  # noqa: BLE001
        # A broken pack must not take the pipeline down with it.
        logger.warning(f"could not read pack {name!r} ({path}): {exc}; "
                       "module defaults apply")
    _cache[name] = data
    return data


def value(dotted_key: str, default: Any = None) -> Any:
    """`value("carousel.questions", QUESTIONS)` — pack wins, default survives.

    A key present but empty (`[]`, `""`, `{}`) is treated as absent. Emptying a
    copy bank in a pack is far more likely to be an editing accident than a
    request for a caption with no question in it, and the cost of guessing wrong
    is a live post.
    """
    node: Any = load()
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    if node is None or node == [] or node == {} or node == "":
        return default
    return node


def typed(dotted_key: str, default: Any) -> Any:
    """Like value(), but also refuses a pack entry of the wrong type.

    A list where a dict belongs would otherwise fail deep inside a renderer, at
    publish time, on an account nobody is watching.
    """
    got = value(dotted_key, default)
    if default is not None and not isinstance(got, type(default)):
        logger.warning(f"pack key {dotted_key!r} is {type(got).__name__}, "
                       f"expected {type(default).__name__}; using the default")
        return default
    return got


def describe() -> dict:
    """What is loaded, for the CLI and for the runbook."""
    data = load()
    return {
        "pack": pack_name(),
        "path": pack_path(),
        "exists": os.path.exists(pack_path()),
        "sections": sorted(data.keys()),
        "brand": data.get("brand", {}),
    }


def available() -> list[str]:
    try:
        return sorted(d for d in os.listdir(PACKS_DIR)
                      if os.path.exists(os.path.join(PACKS_DIR, d, "pack.yaml")))
    except OSError:
        return []


def resolved() -> dict:
    """Every value the pipeline will actually read, and where it came from.

    The point of a pack is that you can tell what an account will do without
    reading eight modules — so this reports the effective value and whether the
    pack or the module default supplied it.
    """
    from app.services import carousel, content_scheduler, hashtags, series, verse_card

    checks = [
        ("brand.wordmark", carousel.wordmark(), "holy ordinary"),
        ("carousel.subjects", len(carousel.subjects()), len(carousel.SUBJECTS)),
        ("carousel.cover_variants", len(typed("carousel.cover_variants",
                                              carousel.COVER_VARIANTS)),
         len(carousel.COVER_VARIANTS)),
        ("hashtags.sets", len(hashtags._sets()), len(hashtags.SETS)),
        ("captions.questions", len(typed("captions.questions", hashtags.QUESTIONS)),
         len(hashtags.QUESTIONS)),
        ("verse_card.background_subjects", len(verse_card.background_subjects()),
         len(verse_card.BACKGROUND_SUBJECTS)),
        ("series", len(series.all_series()), len(series.SERIES)),
        ("cadence", content_scheduler.cadence(), content_scheduler.PLAN),
    ]
    data = load()
    out = {}
    for key, effective, default in checks:
        root = key.split(".")[0]
        node: Any = data
        present = True
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                present = False
                break
            node = node[part]
        out[key] = {
            "effective": effective,
            "from": "pack" if present else "module default",
            "matches_default": effective == default,
        }
        del root
    return out


def main() -> int:
    import json
    import sys

    if "--resolved" in sys.argv[1:]:
        print(json.dumps(resolved(), indent=2, default=str))
        return 0
    info = describe()
    info["available"] = available()
    print(json.dumps(info, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
