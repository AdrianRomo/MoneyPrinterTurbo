# Feed variety — what changed, and how to revert it

**Date:** 2026-08-21 · **Branch:** `feat/content-quality-and-reuse` · **Account:** `@faith.intheordinary` (pack `holy-ordinary`)

## The problem

Posts published 2026-08-14..27 all looked the same. Measured over the 41 cards:

| Axis | Before | Meaning |
|---|---|---|
| Saturation | never above 33/100 | 100% "muted", no exceptions |
| Lightness | never above 54/100 | **nothing bright, ever** |
| Palette buckets | 7 total, top one 41% of feed | two looks carried the account |
| Neighbouring posts sharing a bucket | 56% | the scroll repeats |
| Distinct looks in the visible 3×3 grid | 2 | the profile reads as one post |

Only 5 of 50 cards were near-duplicate *pixels*, so nothing that compares files
would have caught this. The sameness was stylistic.

## Four causes, all fixed

1. **One style string on every image ever generated.** `STYLE_SUFFIX` was
   appended to every prompt — "muted warm tones, cinematic, 35mm". A hard cap on
   variety, applied globally.
2. **Background subject picked by `random.choice`.** The only selection axis in
   the pipeline with no anti-repeat; verse references and carousel subjects both
   had one. Random *with replacement* over 30 near-synonyms guarantees clumping.
3. **A 30-item pool that was one genre.** All unpeopled soft nature landscape;
   "light" appeared in 12 of 30. No macro, no architecture, no still life.
4. **The scrim made a light card impossible.** This is the big one, and it is
   invisible from the prompt side. `compose_card` darkens adaptively to keep
   WHITE type legible, so a high-key background composes to a dark card anyway.
   Verified end to end: six backgrounds spanning light, mid and dark all
   composed to "dark". **Widening the vocabulary alone could never have fixed
   the lightness axis.**

## What changed

| File | Change |
|---|---|
| `app/services/rotation.py` | **New.** Shared least-recently-used rotation. Carousel's fix from `f3f1ba2`, extracted — the same bug was live on two other axes. |
| `app/services/verse_card.py` | Subject pool 30 → **115** across 11 registers; `STYLE_SUFFIX` → `STYLE_CORE` + **8 rotating looks**; LRU subject and look selection; `subject_class()`; `generate_background_tagged()`; **`ink` polarity** in `compose_card`; rejected candidates cleaned up. |
| `app/services/quality.py` | **`check_variety()`** — the first gate that compares a card to the *previous* ones. Ink-aware `contrast_ratio` / `alpha_for_target` / `check_card`. |
| `app/services/brand_footage.py` | Cache key now includes the look. Without this the cache defeated rotation: the first look a subject was generated under became its permanent appearance (which is why the Reel path reused four stills). |
| `app/services/carousel.py` | Delegates to `rotation.py`. Fixed a latent `NameError` in `choose_subject()`'s empty-pool fallback. |
| `packs/holy-ordinary/pack.yaml` | Regenerated from the module constants. |
| `scripts/feed_variety_report.py` | **New.** The squint test, automated. |

### The two gates are not equal

- **Contrast is a floor.** Illegible cards are never published. Unchanged.
- **Variety is a preference.** A repetitive card is worth far more than a missed
  slot on an unattended account, so `create_card` tries 4 backgrounds and
  publishes the best legible one even if all 4 repeat. Do not promote it to a
  hard floor without giving the caller somewhere to fall back.

## Result (measured, not estimated)

Six backgrounds generated live through ComfyUI and composed:

| | Before | After |
|---|---|---|
| Distinct palettes | 2 in the last 9 | **6 of 6** |
| Neighbours sharing a bucket | 56% | **0%** |
| Top bucket share | 41% | **17%** |
| Light-band cards possible | **no — 0 of 41** | **yes** |
| Composed contrast | 4.0:1 | 4.0–16.0:1 (dark-ink cards are the *most* legible) |

## How to revert

Everything is pack-editable; no code change is needed to undo the visible parts.

- **Back to all-white type** (the single change that alters the account's visual
  identity): set every `ink` to `light` in `packs/holy-ordinary/pack.yaml`.
- **Back to the original single look:** delete the `style_looks` key from the
  pack. `style_for()` falls back to `STYLE_SUFFIX`, which is unchanged and is
  byte-for-byte the string that shipped.
- **Back to the original 30 subjects:** restore `background_subjects` in the pack
  from git history. The pack overrides the module.
- **Disable the variety gate only:** set `quality.VARIETY_WINDOW = 0`.
- **Full revert:** `git checkout -- app/services packs test` and delete
  `app/services/rotation.py`, `scripts/feed_variety_report.py`.

Rotation state is advisory and safe to delete — a missing file means "nothing
used yet":

    storage/verse_cards/used_backgrounds.json
    storage/verse_cards/used_looks.json
    storage/verse_cards/used_registers.json

## Monitoring

    python scripts/feed_variety_report.py             # both surfaces
    python scripts/feed_variety_report.py --kind post # the grid a visitor sees

Posts and stories are measured **separately**: a story twin deliberately reuses
its feed card's verse and background, so pooling them reports the twin as a
variety failure when it is a feature.

Watch for: top bucket above ~30%, or fewer than 5 distinct looks in the last 9.

## Not done

- **Instagram insights are not yet fed back into look selection.** `insights.py`
  already pulls per-post metrics; correlating engagement against look and
  register would let the winners bias rotation. Left as the next step.
- **`local_videos` holds only 4 brand stills.** The cache-key fix means new ones
  will accumulate under different looks, but the existing four are unchanged.
- **Published posts were not touched.** Nothing was deleted or re-published.
