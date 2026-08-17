import json
import os.path
import re
from timeit import default_timer as timer

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None
from loguru import logger

from app.config import config
from app.services import subtitle_cadence
from app.utils import utils

model = None
_model_key = None


def _settings() -> tuple[str, str, str]:
    # Read at call time, not import time: values baked at import survive a
    # config edit until the process restarts, which is the stale-config trap
    # this pipeline has been bitten by before.
    return (
        str(config.whisper.get("model_size", "large-v3")),
        str(config.whisper.get("device", "cpu")),
        str(config.whisper.get("compute_type", "int8")),
    )


def _load_model(model_size: str, device: str, compute_type: str):
    """Load Whisper, falling back to CPU if the GPU will not have it.

    The 3090 is shared with Ollama and the reranker, so a CUDA load can fail on
    VRAM at any time. Losing captions entirely over that is a far worse outcome
    than transcribing slowly, so the fallback is automatic and loud.
    """
    model_path = f"{utils.root_dir()}/models/whisper-{model_size}"
    if not os.path.isfile(f"{model_path}/model.bin"):
        model_path = model_size

    attempts = [(device, compute_type)]
    if device != "cpu":
        attempts.append(("cpu", "int8"))

    last_error = None
    for attempt_device, attempt_compute in attempts:
        logger.info(
            f"loading model: {model_path}, device: {attempt_device}, "
            f"compute_type: {attempt_compute}"
        )
        try:
            return WhisperModel(
                model_size_or_path=model_path,
                device=attempt_device,
                compute_type=attempt_compute,
            )
        except Exception as e:  # noqa: BLE001 - try CPU before giving up
            last_error = e
            if attempt_device != "cpu":
                logger.warning(
                    f"whisper failed to load on {attempt_device} ({e}); falling back to cpu"
                )

    logger.error(
        f"failed to load model: {last_error} \n\n"
        f"********************************************\n"
        f"this may be caused by network issue. \n"
        f"please download the model manually and put it in the 'models' folder. \n"
        f"see [README.md FAQ](https://github.com/AdrianRomo/Influencer-Automation-2.0) for more details.\n"
        f"********************************************\n\n"
    )
    return None


def create(audio_file, subtitle_file: str = ""):
    global model, _model_key
    if WhisperModel is None:
        logger.warning("faster_whisper not available, skipping whisper subtitle generation")
        return ""

    model_size, device, compute_type = _settings()
    key = (model_size, device, compute_type)
    if model is None or _model_key != key:
        model = _load_model(model_size, device, compute_type)
        _model_key = key if model is not None else None
        if model is None:
            return None

    logger.info(f"start, output file: {subtitle_file}")
    if not subtitle_file:
        subtitle_file = f"{audio_file}.srt"

    segments, info = model.transcribe(
        audio_file,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    logger.info(
        f"detected language: '{info.language}', probability: {info.language_probability:.2f}"
    )

    start = timer()
    subtitles = []

    if subtitle_cadence.enabled():
        # Word-level cues: the whole point of running Whisper rather than
        # reusing the TTS boundaries, and what keeps a cue inside two lines.
        words = [word for segment in segments for word in (segment.words or [])]
        subtitles = subtitle_cadence.group(words)
        logger.info(subtitle_cadence.report(subtitles))
        _write_srt(subtitles, subtitle_file)
        logger.info(f"complete, elapsed: {timer() - start:.2f} s")
        return

    def recognized(seg_text, seg_start, seg_end):
        seg_text = seg_text.strip()
        if not seg_text:
            return

        msg = "[%.2fs -> %.2fs] %s" % (seg_start, seg_end, seg_text)
        logger.debug(msg)

        subtitles.append(
            {"msg": seg_text, "start_time": seg_start, "end_time": seg_end}
        )

    for segment in segments:
        words_idx = 0
        words_len = len(segment.words)

        seg_start = 0
        seg_end = 0
        seg_text = ""

        if segment.words:
            is_segmented = False
            for word in segment.words:
                if not is_segmented:
                    seg_start = word.start
                    is_segmented = True

                seg_end = word.end
                # If it contains punctuation, then break the sentence.
                seg_text += word.word

                if utils.str_contains_punctuation(word.word):
                    # remove last char
                    seg_text = seg_text[:-1]
                    if not seg_text:
                        continue

                    recognized(seg_text, seg_start, seg_end)

                    is_segmented = False
                    seg_text = ""

                if words_idx == 0 and segment.start < word.start:
                    seg_start = word.start
                if words_idx == (words_len - 1) and segment.end > word.end:
                    seg_end = word.end
                words_idx += 1

        if not seg_text:
            continue

        recognized(seg_text, seg_start, seg_end)

    end = timer()

    diff = end - start
    logger.info(f"complete, elapsed: {diff:.2f} s")

    _write_srt(subtitles, subtitle_file)


def _write_srt(subtitles: list, subtitle_file: str) -> None:
    idx = 1
    lines = []
    for subtitle in subtitles:
        text = subtitle.get("msg")
        if text:
            lines.append(
                utils.text_to_srt(
                    idx, text, subtitle.get("start_time"), subtitle.get("end_time")
                )
            )
            idx += 1

    sub = "\n".join(lines) + "\n"
    with open(subtitle_file, "w", encoding="utf-8") as f:
        f.write(sub)
    logger.info(f"subtitle file created: {subtitle_file}")


def file_to_subtitles(filename):
    if not filename or not os.path.isfile(filename):
        return []

    times_texts = []
    current_times = None
    current_text = ""
    index = 0
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            times = re.findall("([0-9]*:[0-9]*:[0-9]*,[0-9]*)", line)
            if times:
                current_times = line
            elif line.strip() == "" and current_times:
                index += 1
                times_texts.append((index, current_times.strip(), current_text.strip()))
                current_times, current_text = None, ""
            elif current_times:
                current_text += line

    # Flush the final block. SRT files whose last subtitle is not followed by a
    # trailing blank line never hit the blank-line branch above, so without this
    # the last subtitle would be silently dropped.
    if current_times:
        index += 1
        times_texts.append((index, current_times.strip(), current_text.strip()))
    return times_texts


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def similarity(a, b):
    distance = levenshtein_distance(a.lower(), b.lower())
    max_length = max(len(a), len(b))
    return 1 - (distance / max_length)


def correct(subtitle_file, video_script):
    subtitle_items = file_to_subtitles(subtitle_file)
    normalized_script = utils.normalize_script_for_subtitle_matching(video_script)
    script_lines = utils.split_string_by_punctuations(normalized_script)

    corrected = False
    new_subtitle_items = []
    script_index = 0
    subtitle_index = 0

    while script_index < len(script_lines) and subtitle_index < len(subtitle_items):
        script_line = script_lines[script_index].strip()
        subtitle_line = subtitle_items[subtitle_index][2].strip()

        if script_line == subtitle_line:
            new_subtitle_items.append(subtitle_items[subtitle_index])
            script_index += 1
            subtitle_index += 1
        else:
            combined_subtitle = subtitle_line
            start_time = subtitle_items[subtitle_index][1].split(" --> ")[0]
            end_time = subtitle_items[subtitle_index][1].split(" --> ")[1]
            next_subtitle_index = subtitle_index + 1

            while next_subtitle_index < len(subtitle_items):
                next_subtitle = subtitle_items[next_subtitle_index][2].strip()
                if similarity(
                    script_line, combined_subtitle + " " + next_subtitle
                ) > similarity(script_line, combined_subtitle):
                    combined_subtitle += " " + next_subtitle
                    end_time = subtitle_items[next_subtitle_index][1].split(" --> ")[1]
                    next_subtitle_index += 1
                else:
                    break

            if similarity(script_line, combined_subtitle) > 0.8:
                logger.warning(
                    f"Merged/Corrected - Script: {script_line}, Subtitle: {combined_subtitle}"
                )
                new_subtitle_items.append(
                    (
                        len(new_subtitle_items) + 1,
                        f"{start_time} --> {end_time}",
                        script_line,
                    )
                )
                corrected = True
            else:
                logger.warning(
                    f"Mismatch - Script: {script_line}, Subtitle: {combined_subtitle}"
                )
                new_subtitle_items.append(
                    (
                        len(new_subtitle_items) + 1,
                        f"{start_time} --> {end_time}",
                        script_line,
                    )
                )
                corrected = True

            script_index += 1
            subtitle_index = next_subtitle_index

    # Process the remaining lines of the script.
    while script_index < len(script_lines):
        logger.warning(f"Extra script line: {script_lines[script_index]}")
        if subtitle_index < len(subtitle_items):
            new_subtitle_items.append(
                (
                    len(new_subtitle_items) + 1,
                    subtitle_items[subtitle_index][1],
                    script_lines[script_index],
                )
            )
            subtitle_index += 1
        else:
            new_subtitle_items.append(
                (
                    len(new_subtitle_items) + 1,
                    "00:00:00,000 --> 00:00:00,000",
                    script_lines[script_index],
                )
            )
        script_index += 1
        corrected = True

    if corrected:
        with open(subtitle_file, "w", encoding="utf-8") as fd:
            for i, item in enumerate(new_subtitle_items):
                fd.write(f"{i + 1}\n{item[1]}\n{item[2]}\n\n")
        logger.info("Subtitle corrected")
    else:
        logger.success("Subtitle is correct")


if __name__ == "__main__":
    task_id = "c12fd1e6-4b0a-4d65-a075-c87abe35a072"
    task_dir = utils.task_dir(task_id)
    subtitle_file = f"{task_dir}/subtitle.srt"
    audio_file = f"{task_dir}/audio.mp3"

    subtitles = file_to_subtitles(subtitle_file)
    print(subtitles)

    script_file = f"{task_dir}/script.json"
    with open(script_file, "r") as f:
        script_content = f.read()
    s = json.loads(script_content)
    script = s.get("script")

    correct(subtitle_file, script)

    subtitle_file = f"{task_dir}/subtitle-test.srt"
    create(audio_file, subtitle_file)
