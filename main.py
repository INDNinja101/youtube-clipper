"""
Local YouTube -> Shorts clipper.

Pipeline:
  1. Pull metadata + auto-captions for a YouTube video (no full download yet).
  2. If captions are missing, fall back to local Whisper transcription.
  3. Download the video once.
  4. Ask Gemini to pick the strongest standalone moments, scored against the
     rulebook (RULEBOOK.md): hook, emotion, stakes, payoff, broad appeal.
  5. Keep only clips that pass both the score threshold and the rulebook gate.
  6. Snap each clip to word boundaries so it opens on the first word of the
     hook and ends on the payoff.
  7. Ask Gemini (vision) where the speaking faces are, and build a shot plan:
     full-screen speaker crops, stacked split-screen for back-and-forth,
     punch-ins on the punchlines.
  8. Render 1080x1920 with dead air cut out, a light grade, word captions
     coloured per speaker with keywords highlighted, an optional context box
     and a watermark.
  9. Write each clip with a ready-to-paste post kit (title, description,
     hashtags, pinned comment, suggested post time) and record it in a
     history file so the same moment is never rendered twice.

Nothing is uploaded anywhere. Clips land on disk for you to review and post.
"""

from __future__ import annotations

import argparse
import base64
import bisect
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from math import hypot
from pathlib import Path
from typing import Any, Callable

import requests
import yaml

# Run yt-dlp through the Python running this script, so it works whether or
# not the venv is activated and whether or not yt-dlp is on PATH.
YTDLP = [sys.executable, "-m", "yt_dlp"]

GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

OUT_W, OUT_H, OUT_FPS = 1080, 1920, 30


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------


class PipelineError(RuntimeError):
    """Anything that should stop the run with a readable message."""


def log(msg: str) -> None:
    print(f"  {msg}", flush=True)


def step(msg: str) -> None:
    print(f"\n[*] {msg}", flush=True)


def run(cmd: list[str], cwd: Path | None = None, quiet: bool = True) -> str:
    """Run a subprocess, raising a readable error if it fails."""
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise PipelineError(
            f"Could not find '{cmd[0]}'. Is it installed and on your PATH?"
        ) from exc

    name = "yt-dlp" if cmd[:len(YTDLP)] == YTDLP else cmd[0]
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-25:]
        raise PipelineError(
            f"'{name}' failed (exit {proc.returncode}):\n" + "\n".join(tail)
        )
    if not quiet and proc.stdout:
        print(proc.stdout)
    return proc.stdout


def require_tools() -> None:
    missing = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        missing.append("yt-dlp (pip install -r requirements.txt)")
    if missing:
        raise PipelineError(
            "Missing required tool(s): "
            + ", ".join(missing)
            + "\nSee the README for install instructions."
        )


def safe_filename(text: str, max_len: int = 60) -> str:
    """Turn a clip title into something Windows will accept as a filename."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text).strip()
    text = re.sub(r"[\s_-]+", "-", text)
    text = text.strip("-.") or "clip"
    # Windows reserved device names
    if text.upper().split("-")[0] in {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        text = "x-" + text
    return text[:max_len].rstrip("-")


def ass_time(seconds: float) -> str:
    """Format seconds as ASS timestamp: H:MM:SS.cc"""
    total_cs = int(round(max(0.0, seconds) * 100))
    h, rem = divmod(total_cs, 360_000)
    m, rem = divmod(rem, 6_000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def clamp(v: Any, lo: float, hi: float, default: float) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if f != f:  # NaN
        return default
    return max(lo, min(hi, f))


# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------


@dataclass
class Config:
    output_dir: str = "clips"
    work_dir: str = ".clipper-work"
    keep_work_files: bool = False

    max_source_height: int = 1080

    # --- selection (RULEBOOK §5, §10) ---
    min_clip_seconds: int = 15
    max_clip_seconds: int = 60
    target_clip_count: int = 12
    score_threshold: int = 75
    min_gate_points: int = 7
    max_clips_to_render: int = 10
    channel_focus: str = ""
    title_language: str = "English"

    # --- framing and edit (RULEBOOK §6) ---
    layout: str = "speaker"  # "speaker" | "crop" | "blur"
    # How "speaker" layout finds people: "faces" = local face detection (no API
    # calls), "gemini" = Gemini vision. faces falls back to gemini, then centre.
    framing: str = "faces"
    framing_fallback_to_gemini: bool = True
    face_sample_fps: float = 10.0
    vision_frame_interval: float = 1.5
    vision_max_frames: int = 30
    punch_ins: bool = True
    punch_zoom: float = 1.15
    cut_pauses: bool = True
    max_pause_seconds: float = 0.5
    color_grade: bool = True
    # "auto" = GPU (NVENC) when it works, otherwise CPU x264. "nvenc" | "x264" to force.
    video_encoder: str = "auto"

    # --- captions (RULEBOOK §6 C1-C8) ---
    burn_captions: bool = True
    caption_font: str = "Impact"
    caption_font_size: int = 100
    caption_words_per_line: int = 3
    caption_max_chars: int = 16
    caption_vertical_pos: float = 0.62  # 0 = top, 1 = bottom
    caption_language: str = "English"  # translate captions into this; "" = keep original
    speaker_colors: list[str] = field(default_factory=lambda: ["#FFFFFF", "#6FE8FF", "#B8FF6B"])
    keyword_color: str = "#FFE600"
    censor_words: list[str] = field(default_factory=list)

    # --- overlays ---
    hook_text: bool = True
    hook_seconds: float = 2.5
    hook_font: str = "Arial Black"
    watermark: str = ""

    # --- posting (RULEBOOK §13) ---
    post_times: list[str] = field(default_factory=lambda: ["12:30", "17:30", "21:00"])
    allow_repeats: bool = False

    gemini_model: str = "gemini-3.6-flash"
    gemini_api_key_env: str = "GEMINI_API_KEY"
    # tried in order when the main model is overloaded or rate-limited
    gemini_fallback_models: list[str] = field(default_factory=lambda: [
        "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"])

    whisper_model: str = "base"
    whisper_compute_type: str = "int8"

    subtitle_langs: str = "en.*,en"

    @classmethod
    def load(cls, path: Path) -> "Config":
        if not path.exists():
            raise PipelineError(f"Config file not found: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            log(f"note: ignoring unknown config keys: {', '.join(sorted(unknown))}")
        cfg = cls(**{k: v for k, v in raw.items() if k in known and v is not None})
        if cfg.layout not in ("speaker", "crop", "blur"):
            raise PipelineError(f"layout must be speaker, crop or blur (got '{cfg.layout}')")
        if cfg.framing not in ("faces", "gemini"):
            raise PipelineError(f"framing must be faces or gemini (got '{cfg.framing}')")
        if cfg.video_encoder not in ("auto", "nvenc", "x264"):
            raise PipelineError(f"video_encoder must be auto, nvenc or x264 (got '{cfg.video_encoder}')")
        return cfg


# ----------------------------------------------------------------------------
# transcript
# ----------------------------------------------------------------------------


@dataclass
class Word:
    start: float
    end: float
    text: str


NETWORK_FLAGS = ["--retries", "10", "--socket-timeout", "30"]


def run_patiently(cmd: list[str], cwd: Path | None = None, waits: tuple[int, ...] = (5, 15, 30)) -> str:
    """
    run(), retried after short waits. For yt-dlp calls: a DNS hiccup or a read
    timeout is normal on a home connection and nearly always passes.
    """
    for wait in waits:
        try:
            return run(cmd, cwd=cwd)
        except PipelineError as exc:
            log(f"network trouble, retrying in {wait}s: {str(exc).splitlines()[-1][:120]}")
            time.sleep(wait)
    return run(cmd, cwd=cwd)


def fetch_metadata(url: str) -> dict[str, Any]:
    out = run_patiently([*YTDLP, "--dump-single-json", "--no-warnings", "--skip-download",
                         *NETWORK_FLAGS, url])
    return json.loads(out)


def choose_caption_track(meta: dict[str, Any], langs: str) -> tuple[str, bool] | None:
    """
    Pick one caption track as (lang, is_auto).

    Manual subs in a wanted language win. Otherwise use the auto-caption track
    in the language actually spoken ("-orig"), not a machine translation of it:
    translated tracks have poor timing and YouTube 429s them aggressively.
    """
    patterns = [p.strip() for p in langs.split(",") if p.strip()]

    def wanted(lang: str) -> bool:
        return any(re.fullmatch(p, lang) for p in patterns)

    manual = [l for l in (meta.get("subtitles") or {}) if l != "live_chat"]
    for lang in manual:
        if wanted(lang):
            return lang, False

    auto = list(meta.get("automatic_captions") or {})
    orig = [l for l in auto if l.endswith("-orig")]
    spoken = str(meta.get("language") or "")
    if orig:
        # YouTube can list several "original" tracks (it once offered bn-orig,
        # en-US-orig and hi-orig for a Hindi show). Take the one in the
        # language actually spoken; the list order means nothing.
        base = spoken.split("-")[0].lower()
        for lang in orig:
            if base and lang.removesuffix("-orig").split("-")[0].lower() == base:
                return lang, True
        return orig[0], True
    if spoken and spoken in auto:
        return spoken, True
    if manual:
        return manual[0], False
    return None


def fetch_youtube_captions(url: str, work: Path, meta: dict[str, Any],
                           langs: str) -> tuple[list[Word], str]:
    """
    Try to grab YouTube's own caption track. Cheap, instant, no download.
    Returns the words and the track's language code.
    """
    track = choose_caption_track(meta, langs)
    if track is None:
        return [], ""
    lang, is_auto = track
    log(f"using {'auto' if is_auto else 'manual'} caption track '{lang}'")

    for old in work.glob("subs*.json3"):
        old.unlink()
    # The track exists, so a failure here is the network, not a video without
    # captions. Failing loudly beats silently falling back to transcribing an
    # hour of audio (or telling the user to install a transcriber).
    try:
        run_patiently([
            *YTDLP,
            "--skip-download",
            "--write-auto-subs" if is_auto else "--write-subs",
            "--sub-langs", re.escape(lang),
            "--sub-format", "json3",
            "--no-warnings",
            *NETWORK_FLAGS,
            "-o", "subs.%(ext)s",
            url,
        ], cwd=work)
    except PipelineError as exc:
        raise PipelineError(
            f"YouTube has captions for this video ('{lang}') but they could not be "
            f"downloaded:\n{str(exc).splitlines()[-1][:300]}\n"
            "This is a connection problem. Check your internet and run the same command again."
        ) from exc

    files = sorted(work.glob("subs*.json3"))
    if not files:
        return [], ""
    return parse_json3(files[0]), lang.removesuffix("-orig")


def parse_json3(path: Path) -> list[Word]:
    """
    Parse YouTube's json3 caption format into word-level timings.

    Events with aAppend=1 are the rolling duplicates of the previous line,
    so they get dropped or every word appears twice.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    words: list[Word] = []

    for event in data.get("events", []):
        if event.get("aAppend") == 1:
            continue
        segs = event.get("segs")
        if not segs:
            continue
        base = event.get("tStartMs", 0)
        for seg in segs:
            # ">>" is YouTube's speaker-change marker, not a word
            text = re.sub(r"^(?:>>+\s*)+", "", (seg.get("utf8") or "").strip()).strip()
            if not text:
                continue
            start = (base + seg.get("tOffsetMs", 0)) / 1000.0
            words.append(Word(start=start, end=start + 0.4, text=text))

    words.sort(key=lambda w: w.start)
    # Give each word an end time that runs up to the next word.
    for i, w in enumerate(words[:-1]):
        w.end = min(words[i + 1].start, w.start + 1.5)
    return words


def transcribe_locally(video: Path, cfg: Config) -> tuple[list[Word], str]:
    """Fallback when YouTube has no usable caption track."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise PipelineError(
            "No captions found on YouTube and faster-whisper isn't installed.\n"
            "Run: pip install faster-whisper"
        ) from exc

    log(f"transcribing locally with Whisper '{cfg.whisper_model}' (this takes a while)")
    model = WhisperModel(cfg.whisper_model, device="cpu", compute_type=cfg.whisper_compute_type)
    segments, info = model.transcribe(str(video), word_timestamps=True, vad_filter=True)

    words: list[Word] = []
    for seg in segments:
        for w in (seg.words or []):
            token = w.word.strip()
            if token:
                words.append(Word(start=w.start, end=w.end, text=token))
    return words, info.language


def transcript_as_lines(words: list[Word], window: float = 8.0) -> str:
    """Group words into timestamped lines for the model to read."""
    if not words:
        return ""
    lines: list[str] = []
    bucket: list[str] = []
    bucket_start = words[0].start

    for w in words:
        if w.start - bucket_start >= window and bucket:
            lines.append(f"[{bucket_start:.1f}] " + " ".join(bucket))
            bucket = []
            bucket_start = w.start
        bucket.append(w.text)

    if bucket:
        lines.append(f"[{bucket_start:.1f}] " + " ".join(bucket))
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# word-level text rules (RULEBOOK §6 C8)
# ----------------------------------------------------------------------------

TAG_RE = re.compile(r"^\[.*\]$|^\(.*\)$")
PROFANITY_TAG = "[ __ ]"  # how YouTube auto-captions mark a bleeped word
SENTENCE_END = ("।", ".", "?", "!", "॥")

# Words the clip must not open on (RULEBOOK §4 H1): the first word on screen
# should already be the hook.
FILLERS = {
    "so", "um", "umm", "uh", "uhh", "hmm", "okay", "ok", "well", "yeah", "and",
    "like", "basically", "actually", "anyway", "right", "toh", "haan", "matlab",
    "acha", "achha", "तो", "हां", "हाँ", "मतलब", "अच्छा",
}

# Never highlighted even if the model hands back a phrase as a "keyword".
KEYWORD_STOPWORDS = {
    "the", "and", "her", "his", "him", "she", "they", "them", "off", "for", "you",
    "your", "that", "this", "with", "was", "are", "not", "but", "had", "has",
    "hai", "tha", "thi", "aur", "nahi", "kya", "है", "था", "और", "नहीं", "क्या",
}

DEFAULT_CENSOR = [
    "fuck", "motherfuck", "shit", "bitch", "cunt", "nigga", "nigger", "faggot",
    "madarchod", "behenchod", "bhenchod", "chutiya", "bhosdi", "gandu",
]


def is_tag(text: str) -> bool:
    """Sound tags like [Music], [हंसी], (laughs). Kept for timing, never shown."""
    return bool(TAG_RE.match(text.strip())) and text.strip() != PROFANITY_TAG


def strip_emoji(text: str) -> str:
    """libass can't draw colour emoji; drop them from anything burned in."""
    return "".join(
        ch for ch in text
        if ord(ch) <= 0xFFFF and unicodedata.category(ch) != "So"
        and not 0xFE00 <= ord(ch) <= 0xFE0F and ch != "\u200d"
    ).strip()


def censor(text: str, bad: list[str]) -> str:
    low = text.lower()
    for b in bad:
        if b and low.lstrip("\"'([").startswith(b):
            core = re.sub(r"\W", "", text)
            return (core[:1] + "*" * max(2, len(core) - 1)) if core else "***"
    return text


def display_text(word: Word, bad: list[str]) -> str:
    """What a word looks like on screen: tags dropped, bleeps and slurs starred."""
    t = word.text.strip()
    if t == PROFANITY_TAG:
        return "****"
    if is_tag(t):
        return ""
    t = re.sub(r">>+", " ", t)  # speaker-change markers that survive translation
    t = strip_emoji(t.replace("\\", "").replace("{", "(").replace("}", ")"))
    return censor(t, bad) if t else ""


def norm_token(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.lower())


def est_speech_end(w: Word) -> float:
    """
    When a word's sound actually stops. YouTube json3 only gives start times
    (end is stretched to the next word), so estimate from the word length.
    """
    if is_tag(w.text):
        guess = 1.2  # laughter/music tags mark the start of a longer sound
    else:
        guess = min(0.9, max(0.18, 0.055 * len(w.text) + 0.12))
    return min(w.end, w.start + guess)


@dataclass
class Line:
    """One sentence (or breath) of the transcript. Clips are chosen in whole lines."""
    id: int
    start: float
    end: float
    text: str


def transcript_sentences(words: list[Word], max_words: int = 18, max_span: float = 8.0,
                         pause: float = 0.7) -> list[Line]:
    """
    Split the transcript into numbered sentences. A sentence ends at sentence
    punctuation, at a pause, or when it gets too long. A laugh/applause tag
    right after a sentence is attached to that sentence, because the reaction
    is part of the punchline (RULEBOOK §5 S3).

    The model picks clips by line number, so a clip can only ever start at the
    start of a sentence and end at the end of one.
    """
    lines: list[Line] = []
    bucket: list[Word] = []

    def flush() -> None:
        if bucket:
            lines.append(Line(len(lines), bucket[0].start, est_speech_end(bucket[-1]),
                              " ".join(w.text for w in bucket)))
            bucket.clear()

    closing = False  # the sentence is over; only trailing reaction tags may still join
    for i, w in enumerate(words):
        nxt = words[i + 1] if i + 1 < len(words) else None
        tag_follows = (nxt is not None and is_tag(nxt.text)
                       and nxt.start - est_speech_end(w) < 2.0)
        bucket.append(w)
        if closing or (
                nxt is None
                or w.text.rstrip().endswith(SENTENCE_END)
                or nxt.start - est_speech_end(w) > pause
                or len(bucket) >= max_words
                or nxt.start - bucket[0].start > max_span):
            if tag_follows:
                closing = True  # keep the laugh/applause with this sentence
                continue
            forced = (not closing and nxt is not None
                      and not w.text.rstrip().endswith(SENTENCE_END)
                      and nxt.start - est_speech_end(w) <= pause)
            if forced and len(bucket) > 6:
                # Unpunctuated speech that simply got too long: break at the
                # longest breath among the recent words instead of mid-phrase.
                gaps = [(bucket[j].start - est_speech_end(bucket[j - 1]), j)
                        for j in range(4, len(bucket))]
                gap, j = max(gaps)
                if gap >= 0.12:
                    rest = bucket[j:]
                    del bucket[j:]
                    flush()
                    bucket.extend(rest)
                    continue
            flush()
            closing = False
    flush()
    return lines


def sentences_for_prompt(lines: list[Line]) -> str:
    return "\n".join(f"L{l.id} [{l.start:.1f}s] {l.text}" for l in lines)


# ----------------------------------------------------------------------------
# clip selection
# ----------------------------------------------------------------------------

# RULEBOOK §10 scoring gate. Points per criterion; publish only at >= min_gate_points.
GATE_POINTS = {
    "sound_off_3s": 2,
    "emotion_peak": 2,
    "crossover": 1,
    "stakes": 1,
    "clean_payoff": 1,
    "broad_appeal": 1,
    "fits_focus": 1,
}
# The 10th point ("trending, under 48h old") is awarded from the upload date.

GENERIC_HASHTAGS = {
    "#viral", "#viralvideo", "#viralshorts", "#trending", "#trendingreels",
    "#fyp", "#foryou", "#foryoupage", "#shortvideo", "#reels", "#explore",
    "#youtubeshorts", "#ytshorts", "#shortsfeed", "#viralreels",
}


@dataclass
class Clip:
    start: float
    end: float
    title: str
    hook: str
    score: int
    reason: str
    caption: str = ""
    file: str = ""
    trigger: str = ""
    gate: dict[str, bool] = field(default_factory=dict)
    gate_points: int = 0
    hook_text: str = ""
    keywords: list[str] = field(default_factory=list)
    keep_pauses: bool = False
    hashtags: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    pinned_comment: str = ""
    punch_ins: list[float] = field(default_factory=list)  # source seconds
    start_line: int = -1  # transcript sentence numbers, when chosen by line
    end_line: int = -1
    post_at: str = ""
    people: list[dict[str, str]] = field(default_factory=list)
    shots: list[dict[str, Any]] = field(default_factory=list)
    kept_ranges: list[list[float]] = field(default_factory=list)
    final_seconds: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start


SELECTION_PROMPT = """\
You pick moments from a long video to post as YouTube Shorts on a faceless clip
channel. You follow a rulebook built from the channels that get 10M-300M views.

Video title: {title}
Channel: {channel}
{focus_line}

WHAT WINS (in order of proven reach):
- wholesome / loyalty / kindness payoffs
- conflict, roasts, someone getting humbled
- cringe or ignorance moments
- celebrity collisions (two famous people in one moment)
- big numbers, money, bets, stakes
- absurd, chaotic moments
- milestones, comebacks
- dark curiosity (crime, secrets, horror, "the truth about...") and myth-busting
WHAT FAILS: inside jokes that need backstory, generic advice with no emotion,
rambling setups, anything "funny if you were there".

THE TRANSCRIPT is split into numbered lines, one sentence each: "L<number>
[<start time>s] <text>". Tags such as [laughter] or [applause] show how the room
reacted. You choose a clip by giving its first and last line. A clip is always
whole lines, so it can never start or stop in the middle of a sentence.
Copy the text of your first and last line exactly into "hook" and "last_line":
the clip is located by that text, and the line numbers are only a cross-check.

EACH CLIP IS A COMPLETE SMALL STORY: setup -> build -> punchline -> reaction.
- start_line is the SETUP: the question, claim, challenge or premise that the
  rest depends on. A viewer who heard nothing before it must understand it.
  If that line says "this", "that", "it", "he" or "there" about something
  mentioned earlier, start one or two lines earlier, where the thing is named.
  If the person answering is reacting to a question, the question is the start.
  Do not start on filler ("so", "um", "okay so") or on the tail of the
  previous topic.
- end_line is the line where the PUNCHLINE or payoff finishes, including the
  answer to any question that was asked and the laugh or reaction that follows.
  Read the next two lines after your end_line: if they finish the thought,
  deliver the punchline, or are the reaction, your end_line is too early.
  Never end on a question, on a line that leads into the next thing, or after
  the topic has already changed.
- ONE idea: one moment, one joke, one story. Stop before the topic changes.
- be understandable in the first 3 seconds with the sound OFF
- length {min_s}-{max_s} seconds, measured as (start time of the line after
  end_line) minus (start time of start_line). The best clips are 20-45
  seconds. Never cut the setup or the payoff to make it shorter; choose a
  different moment instead.

Return the {count} strongest moments. Score each 0-100 on how likely it is to hold
attention on Shorts. Be harsh: most of a transcript is not clip material and a
score above 85 should be rare.

For each moment also judge these, honestly, as true/false:
- sound_off_3s: reading only the burned-in captions of the first line or two
  (every clip has captions), a viewer understands what is going on
- emotion_peak: there is a clear laugh / shock / "aww" / anger peak
- crossover: a second well-known person (a celebrity guest, a famous co-host
  or panelist) is part of the moment
- stakes: money, a challenge, a conflict or a loss is on the line
- clean_payoff: it ends on a punchline or payoff, not mid-thought
- broad_appeal: someone who has never heard of these people would still watch
- fits_focus: it fits the channel focus above (true if there is no focus)

PACKAGING, written in {title_language}:
- title: 3-8 words. Start with the name of the best-known person who appears
  IN the clip, then a strong verb and the surprising outcome. End with 1-2
  emojis that match the emotion (funny/cringe 😭🤣, wholesome 🥹❤️‍🩹, crazy/aura
  💀🔥, shock 😱🤯). CAPS on at most 2 words. Build curiosity without giving
  away the payoff. The title may only promise something that actually happens
  between start_line and end_line: never name a person or a reaction that is
  not in those lines. No hashtags, correct spelling and apostrophes.
- description: one sentence of context for the viewer.
- hashtags: 2-5 specific hashtags (the people's names, the show, the topic).
  Never generic ones like #viral #viralvideo #trending #fyp #shortvideo.
- tags: 5-10 search keywords (name variants, "<name> funny moments", show name).
- pinned_comment: a short question that makes people argue or share an opinion.

EDITING HINTS:
- hook_text: "" for most clips. Give one ONLY when the opening line alone
  leaves a viewer unsure who these people are or what is going on: then a 3-5
  word context line for the top of the screen (plain text, no emoji) in
  {caption_language_label}.
- keywords: 1-4 SINGLE words to highlight in the captions: a name, a number,
  money, the insult or the punchline word. Write each in
  {caption_language_label}, exactly as it will appear in the captions.
- keep_pauses: true only if a silence is itself the joke (an awkward pause
  before the punchline). Otherwise false.
- punch_lines: 1-3 line numbers where the punchline or the biggest reaction
  lands. The edit zooms in briefly there.

Return ONLY a JSON array, no prose and no markdown fences. Each element:
{{
  "start_line": <line number of the setup>,
  "hook": "<the full text of start_line, copied exactly>",
  "end_line": <line number where the payoff and reaction finish>,
  "last_line": "<the full text of end_line, copied exactly>",
  "score": <integer 0-100>,
  "trigger": "<wholesome|conflict|cringe|crossover|money|absurd|milestone|dark_curiosity|advice|other>",
  "gate": {{"sound_off_3s": <bool>, "emotion_peak": <bool>, "crossover": <bool>,
            "stakes": <bool>, "clean_payoff": <bool>, "broad_appeal": <bool>,
            "fits_focus": <bool>}},
  "reason": "<one sentence: why this works or does not>",
  "title": "<title>",
  "description": "<one sentence>",
  "hashtags": ["#..."],
  "tags": ["..."],
  "pinned_comment": "<question>",
  "hook_text": "<context line or empty>",
  "keywords": ["..."],
  "keep_pauses": <bool>,
  "punch_lines": [<line number>]
}}

TRANSCRIPT:
{transcript}
"""


# Models that answered "daily quota used up" during this run; not asked again.
_gemini_out_for_today: set[str] = set()


def ask_gemini_json(prompt: str, cfg: Config, extra_parts: list[dict] | None = None,
                    temperature: float = 0.4) -> Any:
    """Send one prompt (plus optional image parts) to Gemini and return parsed JSON."""
    api_key = os.environ.get(cfg.gemini_api_key_env, "").strip()
    if not api_key:
        raise PipelineError(
            f"Environment variable {cfg.gemini_api_key_env} is not set.\n"
            "Get a free key at https://aistudio.google.com/apikey then see the README."
        )

    body = {
        "contents": [{"parts": [{"text": prompt}, *(extra_parts or [])]}],
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json",
            # Thinking tokens count against this on newer models, so leave
            # plenty of room or long replies get cut off mid-JSON.
            "maxOutputTokens": 32768,
        },
    }

    # Models are tried in order: the main one, then each fallback. A model whose
    # DAILY free quota is used up is dropped for the rest of the run. Per-minute
    # limits (429) and "high demand" (5xx) pass within minutes, so after every
    # model has been tried the round repeats with a growing wait.
    models = [cfg.gemini_model, *(m for m in cfg.gemini_fallback_models if m != cfg.gemini_model)]
    delays = [10, 30, 60]
    resp = None
    for attempt in range(len(delays) + 1):
        for model in models:
            if model in _gemini_out_for_today:
                continue
            try:
                resp = requests.post(
                    GEMINI_ENDPOINT.format(model=model),
                    headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                    json=body,
                    timeout=180,
                )
            except requests.RequestException as exc:
                log(f"{model}: could not connect ({type(exc).__name__})")
                continue
            if resp.status_code != 429 and resp.status_code < 500:
                break
            if resp.status_code == 429 and "PerDay" in resp.text:
                _gemini_out_for_today.add(model)
                log(f"{model}: daily free quota used up (resets at midnight Pacific), skipping it")
            elif resp.status_code == 429:
                log(f"{model}: rate limited for the moment (429)")
            else:
                log(f"{model}: busy ({resp.status_code})")
        else:
            if all(m in _gemini_out_for_today for m in models):
                raise PipelineError(
                    "Every Gemini model in config.yaml has used up its daily free quota.\n"
                    "Quotas reset at midnight Pacific time. Try again after that, or add\n"
                    "another model to gemini_fallback_models."
                )
            if attempt < len(delays):
                log(f"no model answered, retrying in {delays[attempt]}s")
                time.sleep(delays[attempt])
            continue
        if model != cfg.gemini_model:
            log(f"answered by fallback model {model}")
        break

    if resp is None:
        raise PipelineError("Could not reach Gemini. Check your internet connection.")
    if resp.status_code == 429:
        raise PipelineError(
            "Gemini is rate-limiting every model right now. Wait a few minutes and retry;\n"
            "the downloaded video is kept, so the retry is quick."
        )
    if resp.status_code >= 500:
        raise PipelineError(
            "Gemini is overloaded (high demand) on every model. Wait a few minutes and retry;\n"
            "the downloaded video is kept, so the retry is quick."
        )
    if not resp.ok:
        raise PipelineError(f"Gemini returned {resp.status_code}: {resp.text[:500]}")

    payload = resp.json()
    try:
        candidate = payload["candidates"][0]
        parts = candidate["content"]["parts"]
    except (KeyError, IndexError) as exc:
        raise PipelineError(f"Unexpected Gemini response shape: {json.dumps(payload)[:500]}") from exc
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))

    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise PipelineError(
            f"Gemini did not return valid JSON (finishReason={candidate.get('finishReason')}, "
            f"usage={payload.get('usageMetadata')}):\n{text[:500]}\n...\n{text[-300:]}"
        ) from exc


def clean_hashtags(tags: Any) -> list[str]:
    out: list[str] = []
    for t in tags if isinstance(tags, list) else []:
        t = "#" + re.sub(r"[\s#]+", "", str(t))
        if len(t) > 1 and t.lower() not in GENERIC_HASHTAGS and t.lower() not in (o.lower() for o in out):
            out.append(t)
    return out[:5]


def clean_title(title: str) -> str:
    title = re.sub(r"#\S+", "", title)  # hashtags belong in the description
    return re.sub(r"\s+", " ", title).strip() or "clip"


def gate_points(gate: dict[str, bool], trending: bool) -> int:
    return sum(p for k, p in GATE_POINTS.items() if gate.get(k)) + (1 if trending else 0)


QUOTE_MATCH = 0.6  # how closely a quoted line must match a transcript line to be trusted


def _tokens(text: str) -> list[str]:
    return [t for t in (norm_token(x) for x in text.split() if not is_tag(x)) if t]


def locate_line(quote: str, lines: list[Line], near: int | None = None,
                lo: int = 0, hi: int | None = None) -> tuple[int, float]:
    """
    Find the transcript line a verbatim quote came from: (line index, score).
    The model copies text accurately but reports line numbers that can be off
    by hundreds, so the text decides and `near` (its claimed number) only
    breaks ties between equally good matches, such as a repeated phrase.
    """
    want = _tokens(quote)
    if not want:
        return -1, 0.0
    best_i, best = -1, 0.0
    hi = len(lines) if hi is None else min(hi, len(lines))
    for line in lines[max(0, lo):hi]:
        have = _tokens(line.text)
        if not have:
            continue
        matcher = difflib.SequenceMatcher(None, want, have, autojunk=False)
        # the quote may be only the first part of a long line
        covered = sum(b.size for b in matcher.get_matching_blocks()) / len(want)
        score = max(matcher.ratio(), 0.92 * covered if len(want) >= 3 else 0.0)
        closer = (near is not None and best_i >= 0
                  and abs(line.id - near) < abs(best_i - near))
        if score > best + 0.02 or (abs(score - best) <= 0.02 and closer):
            best_i, best = line.id, score
    return best_i, best


def _as_int(v: Any) -> int | None:
    try:
        return int(str(v).strip().lstrip("Ll"))
    except (TypeError, ValueError):
        return None


def parse_candidates(items: Any, trending: bool, lines: list[Line] | None = None) -> list[Clip]:
    """
    Turn the model's reply into clips. With `lines`, a clip is whole sentences:
    its first and last lines are found by their quoted text ("hook" and
    "last_line"), falling back to the start_line/end_line numbers. A reply in
    raw seconds ("start"/"end") is still accepted.
    """
    if not isinstance(items, list):
        raise PipelineError(f"Expected a JSON array from Gemini, got:\n{json.dumps(items)[:500]}")
    clips: list[Clip] = []
    moved = 0
    for it in items:
        if not isinstance(it, dict):
            continue
        si = ei = -1
        punches: list[float] = []
        if lines and ("start_line" in it or "hook" in it) and "start" not in it:
            said_s, said_e = _as_int(it.get("start_line")), _as_int(it.get("end_line"))
            found, score = locate_line(str(it.get("hook") or ""), lines, near=said_s)
            if score >= QUOTE_MATCH:
                si = found
            elif said_s is not None and 0 <= said_s < len(lines):
                si = said_s
            else:
                continue  # neither the quote nor the number points anywhere real
            shift = si - said_s if said_s is not None else 0
            moved += 1 if shift else 0

            found, score = locate_line(str(it.get("last_line") or ""), lines,
                                       near=(said_e + shift) if said_e is not None else None,
                                       lo=si, hi=si + 80)
            if score >= QUOTE_MATCH:
                ei = found
            elif said_e is not None and said_s is not None and said_e >= said_s:
                ei = min(len(lines) - 1, said_e + shift)  # same length as the model meant
            else:
                ei = si
            start, end = lines[si].start, lines[ei].end
            for v in it.get("punch_lines") if isinstance(it.get("punch_lines"), list) else []:
                k = _as_int(v)
                k = k + shift if k is not None else -1
                at = max(lines[k].start, lines[k].end - 0.9) if si <= k <= ei else -1
                if at >= 0 and at not in punches:  # a punchline lands at the end of its sentence
                    punches.append(at)
        elif "start" in it and "end" in it:
            try:
                start, end = float(it["start"]), float(it["end"])
            except (TypeError, ValueError):
                continue
            punches = [clamp(v, 0, 1e9, -1) for v in
                       (it.get("punch_ins") if isinstance(it.get("punch_ins"), list) else [])]
        else:
            continue
        raw_gate = it.get("gate") if isinstance(it.get("gate"), dict) else {}
        gate = {k: bool(raw_gate.get(k)) for k in GATE_POINTS}
        clips.append(Clip(
            start=start,
            end=end,
            title=clean_title(str(it.get("title", "clip"))),
            hook=str(it.get("hook", "")),
            score=int(clamp(it.get("score"), 0, 100, 0)),
            reason=str(it.get("reason", "")),
            caption=str(it.get("description", it.get("caption", ""))),
            trigger=str(it.get("trigger", "")),
            gate=gate,
            gate_points=gate_points(gate, trending),
            hook_text=strip_emoji(str(it.get("hook_text") or "")),
            keywords=[str(k) for k in it.get("keywords") or [] if str(k).strip()][:4],
            keep_pauses=bool(it.get("keep_pauses")),
            hashtags=clean_hashtags(it.get("hashtags")),
            tags=[str(t) for t in it.get("tags") or [] if str(t).strip()][:10],
            pinned_comment=str(it.get("pinned_comment", "")),
            punch_ins=[p for p in punches if start < p < end][:3],
            start_line=si,
            end_line=ei,
        ))
    if moved:
        log(f"{moved} of {len(clips)} clips were located by their quoted text "
            "(the model's line numbers were off)")
    return clips


def select_clips_with_gemini(lines: list[Line], meta: dict[str, Any], cfg: Config,
                             trending: bool) -> list[Clip]:
    focus = cfg.channel_focus.strip()
    prompt = SELECTION_PROMPT.format(
        title=meta.get("title", "untitled"),
        channel=meta.get("channel") or meta.get("uploader") or "unknown",
        focus_line=(
            f"Channel focus: this channel is built around {focus}. Strongly prefer "
            f"moments where {focus} is central." if focus else "Channel focus: none set."
        ),
        count=cfg.target_clip_count,
        min_s=cfg.min_clip_seconds,
        max_s=cfg.max_clip_seconds,
        title_language=cfg.title_language or "English",
        caption_language_label=cfg.caption_language or "the spoken language",
        transcript=sentences_for_prompt(lines),
    )
    return parse_candidates(ask_gemini_json(prompt, cfg), trending, lines)


def snap_to_words(clip: Clip, words: list[Word], tail: float = 0.35,
                  lookback: float = 0.3) -> bool:
    """
    Move the clip's edges onto real word boundaries (RULEBOOK §4 H1, §5 S3):
    open on the first real word (skipping leading filler), close just after
    the last sound in range. Returns False if nothing is spoken in range.

    `lookback` forgives a start time that is slightly late. It must be ~0 for
    a clip that already starts on a sentence, or the last word of the previous
    sentence gets pulled in.
    """
    inside = [w for w in words if clip.start - lookback <= w.start < clip.end]
    spoken = [w for w in inside if not is_tag(w.text)]
    if not spoken:
        return False

    first = 0
    while first < min(2, len(spoken) - 1) and norm_token(spoken[first].text) in FILLERS:
        first += 1
    clip.start = max(0.0, spoken[first].start - 0.08)

    last = inside[-1]
    after = [w for w in words if w.start > last.start]
    end = est_speech_end(last) + tail
    if after:
        end = min(end, after[0].start - 0.05)
    clip.end = max(end, last.start + 0.2)
    return True


def extend_short_clip(clip: Clip, words: list[Word], min_s: float, max_s: float,
                      tail: float = 0.35) -> bool:
    """
    Lengthen a too-short clip forward to the first natural break (a sentence
    end or a pause) after it reaches min_s, so it keeps its hook and gains the
    explanation or reaction after the punchline. Weaker models often return
    just the one-liner. Returns True if the clip now reaches min_s.
    """
    target = clip.start + min_s
    limit = clip.start + max_s
    later = [w for w in words if clip.end - 0.05 <= w.start < limit]
    for i, w in enumerate(later):
        speech_end = est_speech_end(w)
        if speech_end < target:
            continue
        nxt = later[i + 1] if i + 1 < len(later) else None
        pause = nxt is None or nxt.start - speech_end > 0.6
        if w.text.rstrip().endswith(SENTENCE_END) or pause:
            end = speech_end + tail
            if nxt is not None:
                end = min(end, nxt.start - 0.05)
            clip.end = min(end, limit)
            return clip.duration >= min_s
    if later:  # no break before max_s: stop after the last word that fits
        clip.end = min(limit, est_speech_end(later[-1]) + tail)
    return clip.duration >= min_s


MAX_OVERRUN_SECONDS = 15  # a clip may run this far past max_clip_seconds to keep its payoff


def let_reaction_land(clip: Clip, words: list[Word], quiet: float = 1.5,
                      reaction: float = 2.5) -> None:
    """
    Let the ending breathe (RULEBOOK §5 S3): run on through the laugh or the
    beat after the punchline, up to `reaction` seconds when a reaction tag
    follows and `quiet` seconds otherwise, but never into the next person's
    words. If that stretch turns out to be silent, render_clip trims it.
    """
    after = [w for w in words if w.start >= clip.end - 0.06]
    next_spoken = next((w for w in after if not is_tag(w.text)), None)
    limit = next_spoken.start - 0.12 if next_spoken else clip.end + reaction
    tagged = any(is_tag(w.text) and w.start < limit for w in after[:3])
    clip.end = max(clip.end, min(limit, clip.end + (reaction if tagged else quiet)))


def _asks(line: Line) -> bool:
    spoken = " ".join(t for t in line.text.split() if not is_tag(t)).rstrip()
    return spoken.endswith("?")


def include_question_and_answer(clip: Clip, lines: list[Line], hard_max: float,
                                gap: float = 2.5, punctuated: bool = True) -> list[str]:
    """
    Two context rules the model keeps breaking (RULEBOOK §4 H2, §5 S3):
    a clip must not open on an answer whose question is the sentence just
    before it, and must not end on a question whose answer comes right after.
    Widens the clip by whole lines and returns notes on what it did.
    """
    notes: list[str] = []
    prev = lines[clip.start_line - 1] if clip.start_line > 0 else None
    if (prev and _asks(prev) and not _asks(lines[clip.start_line])
            and lines[clip.start_line].start - prev.end <= gap
            and clip.end - prev.start <= hard_max):
        clip.start_line -= 1
        clip.start = prev.start
        notes.append("start moved back to include the question")
    for _ in range(3):
        last = lines[clip.end_line]
        nxt = lines[clip.end_line + 1] if clip.end_line + 1 < len(lines) else None
        if not nxt or nxt.end - clip.start > hard_max:
            break
        spoken = " ".join(t for t in last.text.split() if not is_tag(t)).rstrip()
        reacted = spoken != last.text.rstrip()  # a laugh/applause tag closes the line
        # The speaker carries on: this line is only half a sentence. Only
        # meaningful when the transcript has punctuation to go by.
        unfinished = (punctuated and bool(spoken) and not reacted
                      and not spoken.endswith(SENTENCE_END) and nxt.start - last.end <= 1.6)
        if _asks(last) and nxt.start - last.end <= gap:
            note = "end moved on to include the answer"
        elif unfinished:
            note = "end moved on to finish the sentence"
        else:
            break
        clip.end_line += 1
        clip.end = nxt.end
        if note not in notes:
            notes.append(note)
    return notes


def sanitize_clips(clips: list[Clip], cfg: Config, source_duration: float,
                   words: list[Word] | None = None,
                   lines: list[Line] | None = None) -> tuple[list[Clip], list[Clip]]:
    """
    Drop impossible ranges, fix lengths in whole sentences, snap to words, let
    the ending land, remove overlaps. Returns (kept, dropped); dropped clips
    get the reason prepended.

    A clip chosen by line numbers is never cut inside a sentence: too short is
    fixed by adding the next sentences, and too long is dropped rather than
    chopped, because chopping is what loses the punchline.
    """
    cleaned: list[Clip] = []
    dropped: list[Clip] = []
    hard_max = cfg.max_clip_seconds + MAX_OVERRUN_SECONDS
    # does this transcript mark sentence ends at all? (some auto-captions don't)
    punctuated = bool(lines) and sum(
        1 for l in lines if " ".join(t for t in l.text.split() if not is_tag(t)).endswith(SENTENCE_END)
    ) >= 0.3 * len(lines)

    def drop(c: Clip, why: str) -> None:
        c.reason = f"[{why}] {c.reason}"
        dropped.append(c)

    for c in clips:
        if c.end <= c.start:
            drop(c, "invalid time range")
            continue
        c.start = max(0.0, c.start)
        c.end = min(source_duration, c.end)
        by_line = bool(lines) and 0 <= c.start_line <= c.end_line < len(lines or [])

        if by_line:
            notes = include_question_and_answer(c, lines, hard_max, punctuated=punctuated)
            if notes:
                c.reason = f"[{'; '.join(notes)}] {c.reason}"
            was = c.duration
            while c.duration < cfg.min_clip_seconds and c.end_line + 1 < len(lines):
                c.end_line += 1
                c.end = min(source_duration, lines[c.end_line].end)
            if c.duration - was > 0.5:
                c.reason = f"[extended from {was:.0f}s to {c.duration:.0f}s] {c.reason}"
            if c.duration > hard_max:
                drop(c, f"{c.duration:.0f}s is too long; cutting it would lose the payoff")
                continue
        else:
            if c.duration > hard_max:
                drop(c, f"{c.duration:.0f}s is too long to trim without losing the payoff")
                continue
            if c.duration > cfg.max_clip_seconds:
                c.end = c.start + cfg.max_clip_seconds

        if words and not snap_to_words(c, words, lookback=0.02 if by_line else 0.3):
            drop(c, "no speech in range")
            continue
        c.end = min(source_duration, c.end)

        if words and not by_line and c.duration < cfg.min_clip_seconds:
            was = c.duration
            if extend_short_clip(c, words, cfg.min_clip_seconds, cfg.max_clip_seconds):
                c.end = min(source_duration, c.end)
                c.reason = f"[extended from {was:.0f}s to {c.duration:.0f}s] {c.reason}"
        if words:
            let_reaction_land(c, words)
            c.end = min(source_duration, c.end)

        limit = hard_max + 3 if by_line else cfg.max_clip_seconds + 4
        if c.duration < cfg.min_clip_seconds or c.duration > limit:
            drop(c, f"{c.duration:.0f}s is outside {cfg.min_clip_seconds}-{cfg.max_clip_seconds}s")
            continue
        cleaned.append(c)

    cleaned.sort(key=lambda c: (c.gate_points, c.score), reverse=True)

    kept: list[Clip] = []
    for c in cleaned:
        if any(c.start < k.end and k.start < c.end for k in kept):
            drop(c, "overlaps a stronger clip")
            continue
        kept.append(c)
    return kept, dropped


def is_trending(meta: dict[str, Any], now: datetime | None = None) -> bool:
    """RULEBOOK §10: +1 point when the source is under 48 hours old."""
    now = now or datetime.now()
    ts = meta.get("timestamp") or meta.get("release_timestamp")
    if ts:
        return now - datetime.fromtimestamp(float(ts)) <= timedelta(hours=48)
    day = str(meta.get("upload_date") or "")
    if re.fullmatch(r"\d{8}", day):
        return now - datetime.strptime(day, "%Y%m%d") <= timedelta(days=2)
    return False


# ----------------------------------------------------------------------------
# caption translation
# ----------------------------------------------------------------------------

TRANSLATE_PROMPT = """\
Translate these spoken caption lines from a video titled "{title}" into {language}.
They are casual, often mixed-language speech (e.g. Hindi with English words).
Translate naturally, the way a viewer would want to read subtitles: keep jokes,
slang and insults intact, keep names as they are, and don't add explanations.
Keep it short and punchy; these are burned-in Shorts captions.
Lines with the same "clip" number are consecutive; use them as context.
Drop sound tags like [हंसी], [संगीत], [laughter], [music] and any ">>" markers.
Censor slurs and strong profanity with asterisks, keeping the first letter (f***).

"key" is the one word of your translation worth highlighting on screen: a
name, a number, a money amount, or the punchline word. Most lines have none:
give a key for at most one line in three, and "" for the rest. The key must
be a single word copied exactly from your "text".

Return ONLY a JSON array with one element per input line, same ids:
[{{"id": <id>, "text": "<translation>", "key": "<one word or empty>"}}]

LINES:
{lines}
"""


def caption_lines(words: list[Word], clip: Clip, max_words: int = 12,
                  max_span: float = 4.0, max_gap: float = 0.8) -> list[tuple[float, float, str]]:
    """Group a clip's words into short phrase-sized lines for translation."""
    inside = [w for w in words if w.end > clip.start and w.start < clip.end]
    lines: list[tuple[float, float, str]] = []
    bucket: list[Word] = []

    def flush() -> None:
        if bucket:
            lines.append((bucket[0].start, bucket[-1].end, " ".join(w.text for w in bucket)))
            bucket.clear()

    for w in inside:
        if bucket and (
            len(bucket) >= max_words
            or w.start - bucket[-1].end > max_gap
            or w.start - bucket[0].start > max_span
        ):
            flush()
        bucket.append(w)
        if w.text.endswith(SENTENCE_END):
            flush()
    flush()
    return lines


def spread_words(start: float, end: float, text: str, limit: float) -> list[Word]:
    """
    Give each word of a translated line a slice of the line's time, by length.
    Short lines may stretch to 0.25s a word, but never past `limit`.
    """
    tokens = text.split()
    if not tokens:
        return []
    weights = [len(t) + 1 for t in tokens]
    span = min(max(end - start, 0.25 * len(tokens)), max(limit - start, end - start))
    out: list[Word] = []
    t = start
    for tok, wgt in zip(tokens, weights):
        dur = span * wgt / sum(weights)
        out.append(Word(start=t, end=t + dur, text=tok))
        t += dur
    return out


def needs_translation(source_lang: str, target: str) -> bool:
    """True unless the transcript is already in the caption language (or none is set)."""
    if not target.strip():
        return False
    names = {"english": "en", "hindi": "hi", "spanish": "es", "french": "fr",
             "german": "de", "portuguese": "pt"}
    want = names.get(target.strip().lower(), target.strip().lower())
    return not source_lang.lower().startswith(want)


def translate_captions(clips: list[Clip], words: list[Word], title: str,
                       cfg: Config) -> list[Word]:
    """
    Translate only the transcript lines inside the chosen clips, in one Gemini
    call, and return them as words timed to the original speech.
    """
    lines: list[tuple[int, float, float, str]] = []
    for n, clip in enumerate(clips, start=1):
        for start, end, text in caption_lines(words, clip):
            lines.append((n, start, end, text))
    if not lines:
        return []

    listing = "\n".join(
        json.dumps({"id": i, "clip": n, "text": text}, ensure_ascii=False)
        for i, (n, _, _, text) in enumerate(lines)
    )
    items = ask_gemini_json(
        TRANSLATE_PROMPT.format(title=title, language=cfg.caption_language, lines=listing),
        cfg,
    )
    if not isinstance(items, list):
        raise PipelineError(f"Expected a JSON array from Gemini, got:\n{json.dumps(items)[:500]}")

    translated: dict[int, str] = {}
    keys: dict[int, list[str]] = {}  # clip number -> highlight words, in caption language
    for it in items:
        if not isinstance(it, dict):
            continue
        i = int(clamp(it.get("id"), 0, len(lines) - 1, -1))
        if i < 0:
            continue
        translated[i] = str(it.get("text", ""))
        key = str(it.get("key") or "").strip()
        if key and norm_token(key) in {norm_token(t) for t in translated[i].split()}:
            keys.setdefault(lines[i][0], []).append(key)
    missing = len(lines) - len(translated)
    if missing:
        log(f"warning: {missing} of {len(lines)} caption lines came back untranslated")

    # Keywords chosen from the translation always match the captions. The ones
    # from the selection call are in the transcript's language and would not.
    for n, clip in enumerate(clips, start=1):
        if keys.get(n):
            clip.keywords = list(dict.fromkeys(keys[n]))[:5]

    out: list[Word] = []
    for i, (_, start, end, _) in enumerate(lines):
        limit = lines[i + 1][1] if i + 1 < len(lines) else end + 1.0
        text = re.sub(r"\[[^\]]*\]|\([^)]*\)|>>+", " ", translated.get(i, ""))
        out.extend(spread_words(start, end, text, limit))
    return out


# ----------------------------------------------------------------------------
# shot planning (RULEBOOK §6 V1-V4): who is on screen, where, and how
# ----------------------------------------------------------------------------


@dataclass
class Shot:
    """One stretch of the clip, in seconds from the clip start (source timeline)."""
    start: float
    end: float
    layout: str = "single"  # single | split | blur
    speaker: str = ""
    x: float = 0.5          # face centre, fraction of the source frame
    y: float = 0.4
    zoom: float = 1.0       # 1.0 = full source height
    top: dict[str, Any] | None = None
    bottom: dict[str, Any] | None = None
    punch: bool = False


VISION_PROMPT = """\
You are the camera operator turning a horizontal clip into a vertical 9:16 Short.
Below are frames sampled from the clip, each labelled with its time in seconds,
then the clip's transcript. The clip runs from 0 to {dur:.1f} seconds.

Plan the vertical framing as consecutive segments covering 0 to {dur:.1f}.

Layouts:
- "single": one person fills the screen. Give the centre of their FACE as x,y
  (fractions 0-1 of the full frame width and height) and a zoom: 1.0 shows the
  full frame height; 1.3-2.0 is tighter. Choose the zoom so the head and
  shoulders fill the upper half of a vertical frame: close-ups need ~1.0, a
  desk panel ~1.4-1.7, a person on a far stage ~2.0.
- "split": two people stacked, "top" and "bottom", each with face x,y. Use it
  when two people trade lines quickly (turns under ~3s) or one visibly reacts
  to what the other says, and both appear in the same frames.

Rules:
- Frame whoever is SPEAKING (use the transcript and the mouths). Start a new
  segment when the speaker changes.
- Camera cuts were detected exactly at these times (seconds): {cuts}.
  Faces move at a cut, so every cut is a segment boundary: start a new
  segment there and read positions from the frame just after it. A wide
  shot where people are small needs a higher zoom than a close-up.
- No segment shorter than 1.0 second. Segments are in order with no gaps.
- If nobody is visible (graphics, screen share), use "single" at x=0.5, y=0.5,
  zoom=1.0.
- "punch_ins": up to 4 times (seconds) where a punchline or big reaction lands.
- "people": everyone you framed, with a short id and their name if known,
  otherwise a short description.

Return ONLY JSON:
{{"people": [{{"id": "A", "name": "..."}}],
  "segments": [
    {{"start": 0.0, "end": 3.2, "layout": "single", "speaker": "A", "x": 0.42, "y": 0.35, "zoom": 1.5}},
    {{"start": 3.2, "end": 8.0, "layout": "split", "speaker": "B",
      "top": {{"id": "A", "x": 0.30, "y": 0.40}}, "bottom": {{"id": "B", "x": 0.70, "y": 0.42}}, "zoom": 1.6}}
  ],
  "punch_ins": [12.4]}}

TRANSCRIPT (seconds from clip start):
{transcript}
"""


def probe_video(video: Path) -> tuple[int, int, bool]:
    """(width, height, has_audio) of the source."""
    out = run([
        "ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height",
        "-of", "json", str(video),
    ])
    streams = json.loads(out).get("streams", [])
    vid = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not vid:
        raise PipelineError("The downloaded file has no video stream.")
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    return int(vid["width"]), int(vid["height"]), has_audio


def clean_plan(raw: Any, dur: float, cuts: list[float] | None = None,
               ) -> tuple[list[Shot], list[dict[str, str]], list[float]]:
    """
    Validate the model's shot plan into contiguous, sane segments. Boundaries
    within 0.7s of a detected camera cut are moved onto the cut, because the
    model only sees sampled frames and a crop held across a cut lands on the
    wrong spot.
    """
    raw = raw if isinstance(raw, dict) else {}
    shots: list[Shot] = []
    for s in raw.get("segments") or []:
        if not isinstance(s, dict):
            continue
        start = clamp(s.get("start"), 0, dur, -1)
        end = clamp(s.get("end"), 0, dur, -1)
        if start < 0 or end <= start:
            continue

        def face(d: Any) -> dict[str, Any] | None:
            if not isinstance(d, dict):
                return None
            return {"id": str(d.get("id", "")), "x": clamp(d.get("x"), 0, 1, 0.5),
                    "y": clamp(d.get("y"), 0, 1, 0.4)}

        layout = s.get("layout") if s.get("layout") in ("single", "split") else "single"
        top, bottom = face(s.get("top")), face(s.get("bottom"))
        shot = Shot(start=start, end=end, layout=layout, speaker=str(s.get("speaker", "")),
                    x=clamp(s.get("x"), 0, 1, 0.5), y=clamp(s.get("y"), 0, 1, 0.4),
                    zoom=clamp(s.get("zoom"), 1.0, 2.5, 1.0))
        if layout == "split":
            if top and bottom:
                shot.top, shot.bottom = top, bottom
            else:  # half a split is just a single
                only = top or bottom or {"x": shot.x, "y": shot.y, "id": shot.speaker}
                shot.layout, shot.x, shot.y = "single", only["x"], only["y"]
        shots.append(shot)

    shots.sort(key=lambda s: s.start)
    if not shots:
        shots = [Shot(0.0, dur)]
    for s in shots[1:]:
        near = [c for c in cuts or [] if abs(c - s.start) <= 0.7]
        if near:
            s.start = min(near, key=lambda c: abs(c - s.start))
    # contiguous: each segment starts where the previous one ends
    shots[0].start = 0.0
    for prev, cur in zip(shots, shots[1:]):
        cur.start = prev.end = max(prev.start + 0.01, cur.start)
    shots[-1].end = dur
    shots = [s for s in shots if s.end > s.start]

    # fold anything under 0.8s into its neighbour so the framing doesn't jitter
    merged: list[Shot] = []
    for s in shots:
        if merged and (s.end - s.start) < 0.8:
            merged[-1].end = s.end
        elif merged and (merged[-1].end - merged[-1].start) < 0.8:
            s.start = merged[-1].start
            merged[-1] = s
        else:
            merged.append(s)

    people = [
        {"id": str(p.get("id", "")), "name": str(p.get("name", ""))}
        for p in raw.get("people") or [] if isinstance(p, dict)
    ]
    punches = sorted(
        clamp(p, 0, dur, -1) for p in raw.get("punch_ins") or []
        if 0.5 < clamp(p, 0, dur, -1) < dur - 0.5
    )[:4]
    return merged, people, punches


def apply_punches(shots: list[Shot], punches: list[float], length: float = 1.3) -> list[Shot]:
    """Split shots so each punch-in window is its own piece with punch=True."""
    out = list(shots)
    for p in punches:
        a, b = p - 0.05, p + length
        nxt: list[Shot] = []
        for s in out:
            if s.punch or b <= s.start or a >= s.end:
                nxt.append(s)
                continue
            if s.start < a:
                nxt.append(Shot(**{**asdict(s), "end": a}))
            nxt.append(Shot(**{**asdict(s), "start": max(a, s.start), "end": min(b, s.end), "punch": True}))
            if b < s.end:
                nxt.append(Shot(**{**asdict(s), "start": b}))
        out = nxt
    return [s for s in out if s.end - s.start > 0.05]


def detect_cuts(video: Path, clip: Clip, threshold: float = 0.3) -> list[float]:
    """Camera cuts inside the clip (seconds from clip start), via ffmpeg scene detection."""
    try:
        proc = subprocess.run([
            "ffmpeg", "-hide_banner", "-nostats",
            "-ss", f"{clip.start:.3f}", "-t", f"{clip.duration:.3f}",
            "-i", str(video.resolve()),
            "-vf", f"scale=320:-2,select='gt(scene,{threshold})',showinfo",
            "-an", "-f", "null", "-",
        ], capture_output=True, text=True, encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return []
    cuts = [float(t) for t in re.findall(r"pts_time:([\d.]+)", proc.stderr or "")]
    out: list[float] = []
    for t in sorted(cuts):
        if 0.3 < t < clip.duration - 0.3 and (not out or t - out[-1] > 0.5):
            out.append(round(t, 2))
    return out


def extract_frames(video: Path, clip: Clip, cfg: Config, dest: Path,
                   cuts: list[float]) -> list[tuple[float, bytes]]:
    """Evenly spaced frames, plus one just after every camera cut."""
    interval = max(cfg.vision_frame_interval, clip.duration / max(1, cfg.vision_max_frames))
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{clip.start:.3f}", "-t", f"{clip.duration:.3f}",
        "-i", str(video.resolve()),
        "-vf", f"fps=1/{interval:.3f},scale=640:-2",
        "-q:v", "5", str(dest / "f_%03d.jpg"),
    ])
    frames = [(i * interval, f.read_bytes()) for i, f in enumerate(sorted(dest.glob("f_*.jpg")))]
    for n, cut in enumerate(cuts[:cfg.vision_max_frames]):
        t = min(clip.duration - 0.05, cut + 0.25)
        if any(abs(t - ft) < 0.4 for ft, _ in frames):
            continue
        target = dest / f"c_{n:03d}.jpg"
        run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{clip.start + t:.3f}", "-i", str(video.resolve()),
            "-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "5", str(target),
        ])
        if target.exists():
            frames.append((t, target.read_bytes()))
    return sorted(frames, key=lambda f: f[0])


# ----------------------------------------------------------------------------
# local face framing (RULEBOOK §6 V1-V3, C5). YuNet finds faces, SFace tells
# people apart across camera cuts, and mouth movement shows who is talking.
# Runs on the CPU in seconds and makes no API calls.
# ----------------------------------------------------------------------------

MODELS_DIR = Path(__file__).resolve().parent / "models"
_ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
FACE_MODELS = {
    "detector": ("face_detection_yunet_2023mar.onnx",
                 _ZOO + "face_detection_yunet/face_detection_yunet_2023mar.onnx"),
    "recognizer": ("face_recognition_sface_2021dec.onnx",
                   _ZOO + "face_recognition_sface/face_recognition_sface_2021dec.onnx"),
}
SAME_PERSON_COSINE = 0.363  # OpenCV's recommended SFace match threshold
SINGLE_FACE_HEIGHT = 0.30   # face height as a share of the vertical frame
SPLIT_FACE_HEIGHT = 0.40    # face height as a share of one split-screen half
MIN_CROP_PX = 400           # never crop the source smaller than this (upscale blur)
MOUTH_ACTIVE = 3.0          # mouth movement that counts as talking (quiet faces measure 0-4)
MOUTH_LEAD = 1.3            # how much more the speaker must move than the next face
WIDE_SHOT_FACE = 0.13       # below this face height, a shot counts as a wide shot
CUT_PIXEL_JUMP = 10.0       # a camera cut: mean pixel change between two frames...
CUT_HIST_JUMP = 0.13        # ...together with this much colour-histogram change
SPLIT_TALK_SHARE = 0.35     # share of a shot someone's mouth must move to earn half the screen


def ensure_face_models() -> dict[str, Path]:
    """Face model paths, downloading them from OpenCV's model zoo on first use."""
    MODELS_DIR.mkdir(exist_ok=True)
    paths: dict[str, Path] = {}
    for key, (name, url) in FACE_MODELS.items():
        path = MODELS_DIR / name
        if not path.exists() or path.stat().st_size < 100_000:
            log(f"downloading face model {name} (one time)")
            try:
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
            except requests.RequestException as exc:
                raise PipelineError(f"could not download {name}: {exc}") from exc
            if not resp.content.startswith(b"\x08"):  # ONNX, not an LFS pointer or HTML
                raise PipelineError(f"the download of {name} is not an ONNX model")
            path.write_bytes(resp.content)
        paths[key] = path
    return paths


@dataclass
class FaceTrack:
    """One face followed through one camera shot. Times are seconds from clip start."""
    tid: int
    shot: int
    times: list[float] = field(default_factory=list)
    xs: list[float] = field(default_factory=list)     # face centre, fraction of width
    ys: list[float] = field(default_factory=list)     # face centre, fraction of height
    hs: list[float] = field(default_factory=list)     # face height, fraction of height
    mouth: list[float] = field(default_factory=list)  # mouth movement at each sample
    feats: list[Any] = field(default_factory=list)    # SFace identity vectors
    person: str = ""
    last_mouth: Any = None
    last_eyes: Any = None

    def median(self, values: list[float], a: float = -1e9, b: float = 1e9) -> float:
        inside = sorted(v for t, v in zip(self.times, values) if a <= t < b) or sorted(values)
        return inside[len(inside) // 2]


def face_patches(gray: Any, row: Any, cv2: Any) -> tuple[Any, Any]:
    """Small, brightness-normalised mouth and eye patches from YuNet landmarks."""
    rex, rey, lex, ley, _, _, rmx, rmy, lmx, lmy = (float(v) for v in row[4:14])
    w = float(row[2])

    def patch(cx: float, cy: float, hw: float, hh: float, size: tuple[int, int]) -> Any:
        x0, x1 = max(0, int(cx - hw)), min(gray.shape[1], int(cx + hw))
        y0, y1 = max(0, int(cy - hh)), min(gray.shape[0], int(cy + hh))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        p = cv2.resize(gray[y0:y1, x0:x1], size, interpolation=cv2.INTER_AREA)
        return p - p.mean()

    mw = max(hypot(lmx - rmx, lmy - rmy), 0.25 * w)
    ew = max(hypot(lex - rex, ley - rey), 0.30 * w)
    mouth = patch((rmx + lmx) / 2, (rmy + lmy) / 2 + 0.1 * mw, 0.75 * mw, 0.6 * mw, (32, 24))
    eyes = patch((rex + lex) / 2, (rey + ley) / 2, 0.9 * ew, 0.35 * ew, (32, 12))
    return mouth, eyes


def analyze_faces(video: Path, clip: Clip, cuts: list[float], cfg: Config
                  ) -> tuple[list[FaceTrack], list[int], list[float]]:
    """
    Follow every face through each camera shot. Returns (tracks, sampled
    frames per shot, shot boundaries).

    Camera cuts are found here, on every frame, from the jump in both pixels
    and colour histogram between neighbouring frames. ffmpeg's scene detector
    misses cuts between similar-looking angles (a dark stage), and a missed
    cut leaves the crop aimed at the previous shot's face. Faces are sampled
    at face_sample_fps.
    """
    import cv2
    import numpy as np

    try:  # OpenCV 5 prints harmless DNN backend warnings otherwise
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except AttributeError:
        pass

    paths = ensure_face_models()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise PipelineError("OpenCV could not open the source video")
    stride = max(1, round((cap.get(cv2.CAP_PROP_FPS) or 30.0) / max(1.0, cfg.face_sample_fps)))
    cap.set(cv2.CAP_PROP_POS_MSEC, clip.start * 1000)

    detector = recognizer = None
    tracks: list[FaceTrack] = []
    live: list[FaceTrack] = []
    cut_times: list[float] = []  # time of the first frame of each new shot
    shot_frames = [0]
    shot, n = 0, -1
    prev_thumb = prev_hist = None
    try:
        while cap.grab():
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 - clip.start
            if t >= clip.duration:
                break
            if t < 0:
                continue
            n += 1
            ok, frame = cap.retrieve()
            if not ok:
                continue

            tiny = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
            thumb = cv2.cvtColor(cv2.resize(tiny, (64, 36)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            hist = cv2.calcHist([tiny], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3).flatten()
            hist /= max(float(hist.sum()), 1.0)
            jumped = (prev_thumb is not None
                      and float(np.mean(np.abs(thumb - prev_thumb))) >= CUT_PIXEL_JUMP
                      and float(np.sum(np.abs(hist - prev_hist))) >= CUT_HIST_JUMP)
            since = t - (cut_times[-1] if cut_times else 0.0)
            if t > 0.15 and since > 0.3 and jumped:
                cut_times.append(t)
                shot += 1
                shot_frames.append(0)
                live = []  # a camera cut: faces start over
            prev_thumb, prev_hist = thumb, hist

            if n % stride:
                continue
            # detect on a 640px copy for speed; read mouths and identities from
            # the full frame so small faces in wide shots still have detail
            h0, w0 = frame.shape[:2]
            sw, sh = 640, max(2, round(h0 * 640 / w0))
            scale = w0 / sw
            small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
            if detector is None:
                detector = cv2.FaceDetectorYN.create(str(paths["detector"]), "", (sw, sh), 0.7, 0.3, 50)
                recognizer = cv2.FaceRecognizerSF.create(str(paths["recognizer"]), "")

            s = shot
            shot_frames[s] += 1

            _, faces = detector.detect(small)
            rows = sorted(faces if faces is not None else [], key=lambda r: -float(r[3]))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) if len(rows) else None
            taken: set[int] = set()
            for row in rows:
                x, y, w, h = (float(v) for v in row[:4])
                if h < 0.035 * sh:
                    continue  # far-background faces
                if x < -0.2 * w or x + w > sw + 0.2 * w or y < -0.2 * h or y + h > sh + 0.2 * h:
                    continue  # face cut off by the frame edge
                full = row.copy()
                full[:14] *= scale
                cx, cy = x + w / 2, y + h / 2
                best, best_d = None, 0.8
                for tr in live:
                    if tr.tid in taken or t - tr.times[-1] > 1.0:
                        continue
                    d = hypot(tr.xs[-1] * sw - cx, tr.ys[-1] * sh - cy) / max(w, 1.0)
                    if d < best_d and 0.6 < (h / sh) / tr.hs[-1] < 1.6:
                        best, best_d = tr, d
                if best is None:
                    best = FaceTrack(tid=len(tracks), shot=s)
                    tracks.append(best)
                    live.append(best)
                taken.add(best.tid)

                mouth_p, eyes_p = face_patches(gray, full, cv2)
                move = 0.0
                if (best.times and t - best.times[-1] <= 0.35
                        and mouth_p is not None and best.last_mouth is not None):
                    m = float(np.mean(np.abs(mouth_p - best.last_mouth)))
                    e = (float(np.mean(np.abs(eyes_p - best.last_eyes)))
                         if eyes_p is not None and best.last_eyes is not None else 0.0)
                    move = max(0.0, m - 0.8 * e)  # subtract whole-head movement
                best.times.append(t)
                best.xs.append(cx / sw)
                best.ys.append(cy / sh)
                best.hs.append(h / sh)
                best.mouth.append(move)
                best.last_mouth, best.last_eyes = mouth_p, eyes_p

                if len(best.feats) < 3 and len(best.times) % 5 == 1:
                    try:
                        feat = recognizer.feature(recognizer.alignCrop(frame, full)).flatten()
                        norm = float(np.linalg.norm(feat))
                        if norm > 0:
                            best.feats.append(feat / norm)
                    except cv2.error:
                        pass
    finally:
        cap.release()
    # A shot boundary sits a hair before its first frame, so that frame can
    # never land in the previous shot's piece when times are rounded.
    bounds = [0.0, *(c - 0.005 for c in cut_times), clip.duration]
    return tracks, shot_frames, bounds


def identify_people(tracks: list[FaceTrack]) -> list[dict[str, str]]:
    """
    Group tracks from different shots into people (P1, P2...) by face
    similarity, so each person keeps one caption colour across cuts. P1 is
    whoever is on screen the most.
    """
    import numpy as np

    groups: list[dict[str, Any]] = []
    for tr in sorted(tracks, key=lambda t: -len(t.times)):
        vec = None
        if tr.feats:
            v = np.mean(tr.feats, axis=0)
            vec = v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else None
        match = None
        if vec is not None:
            best_sim = SAME_PERSON_COSINE
            for g in groups:
                if g["centroid"] is None:
                    continue
                # two faces on screen at the same time are never the same person
                if any(o.shot == tr.shot and o.times[0] < tr.times[-1] and tr.times[0] < o.times[-1]
                       for o in g["tracks"]):
                    continue
                sim = float(np.dot(vec, g["centroid"]))
                if sim >= best_sim:
                    best_sim, match = sim, g
        if match is None:
            groups.append({"centroid": vec, "tracks": [tr]})
            continue
        if vec is not None:
            c = match["centroid"] * len(match["tracks"]) + vec
            match["centroid"] = c / np.linalg.norm(c)
        match["tracks"].append(tr)

    groups.sort(key=lambda g: -sum(len(t.times) for t in g["tracks"]))
    people: list[dict[str, str]] = []
    for i, g in enumerate(groups, start=1):
        for tr in g["tracks"]:
            tr.person = f"P{i}"
        people.append({"id": f"P{i}", "name": ""})
    return people


def speaker_runs(tracks: list[FaceTrack], a: float, b: float, step: float = 0.5,
                 min_run: float = 1.5) -> list[tuple[float, float, int]]:
    """
    Who is talking over [a, b): (start, end, index into tracks) runs. The
    talker is the face whose mouth moves clearly more than the others; quiet
    stretches keep the last talker, and runs under min_run are absorbed so
    the framing doesn't flicker.
    """
    biggest = max(range(len(tracks)), key=lambda i: tracks[i].median(tracks[i].hs))
    winners: list[int | None] = []
    grid: list[float] = []
    g = a
    while g < b - 1e-6:
        mid = g + step / 2
        scores = []
        for tr in tracks:
            vals = [m for t, m in zip(tr.times, tr.mouth) if abs(t - mid) <= 0.6]
            scores.append(sum(vals) / len(vals) if vals else 0.0)
        order = sorted(range(len(tracks)), key=lambda i: -scores[i])
        top = order[0]
        second = scores[order[1]] if len(order) > 1 else 0.0
        clear = scores[top] >= MOUTH_ACTIVE and scores[top] >= MOUTH_LEAD * second
        winners.append(top if clear else None)
        grid.append(g)
        g += step

    first = next((w for w in winners if w is not None), biggest)
    runs: list[list[Any]] = []
    prev = first
    for g, w in zip(grid, winners):
        prev = w if w is not None else prev
        s, e = max(a, g), min(b, g + step)
        if runs and runs[-1][2] == prev:
            runs[-1][1] = e
        else:
            runs.append([s, e, prev])

    out: list[list[Any]] = []
    for r in runs:
        if out and r[1] - r[0] < min_run:
            out[-1][1] = r[1]
        elif out and out[-1][1] - out[-1][0] < min_run:
            r[0] = out[-1][0]
            out[-1] = r
        else:
            out.append(r)
        if len(out) > 1 and out[-1][2] == out[-2][2]:
            out[-2][1] = out[-1][1]
            out.pop()
    if out:
        out[0][0], out[-1][1] = a, b
    return [(s, e, w) for s, e, w in out] or [(a, b, biggest)]


def face_geo(tr: FaceTrack, a: float, b: float, target: float, max_zoom: float) -> dict[str, Any]:
    """Crop centre and zoom that make this face `target` of the frame height."""
    h = tr.median(tr.hs, a, b)
    return {"id": tr.person, "x": tr.median(tr.xs, a, b), "y": tr.median(tr.ys, a, b),
            "zoom": round(clamp(target / max(h, 1e-3), 1.0, max_zoom, 1.0), 3)}


def single_shots(tr: FaceTrack, a: float, b: float, max_zoom: float) -> list[Shot]:
    """One full-screen crop on a face; re-aimed every ~2s if the person moves around."""
    xs = [x for t, x in zip(tr.times, tr.xs) if a <= t < b]
    spans = [(a, b)]
    if xs and max(xs) - min(xs) > 0.12 and b - a > 3:
        n = max(2, round((b - a) / 2))
        spans = [(a + i * (b - a) / n, a + (i + 1) * (b - a) / n) for i in range(n)]
    shots: list[Shot] = []
    for sa, sb in spans:
        g = face_geo(tr, sa, sb, SINGLE_FACE_HEIGHT, max_zoom)
        cur = shots[-1] if shots else None
        # Dead zone: keep the current framing unless the person really moved.
        # Re-aiming for every small shift makes the picture twitch.
        if cur and abs(g["x"] - cur.x) < 0.08 and abs(g["y"] - cur.y) < 0.08 \
                and 0.8 < g["zoom"] / cur.zoom < 1.25:
            cur.end = sb
            continue
        shots.append(Shot(sa, sb, layout="single", speaker=tr.person,
                          x=g["x"], y=g["y"], zoom=g["zoom"]))
    return shots


def talk_share(tr: FaceTrack, a: float, b: float) -> float:
    """Share of this face's samples in [a, b) where its mouth is clearly moving."""
    vals = [m for t, m in zip(tr.times, tr.mouth) if a <= t < b]
    return sum(1 for m in vals if m >= MOUTH_ACTIVE) / len(vals) if vals else 0.0


def face_plan(tracks: list[FaceTrack], bounds: list[float], src_h: int,
              src_w: int | None = None) -> list[Shot]:
    """
    Per camera shot (RULEBOOK §6 V1-V3):
      no faces              -> centre crop
      one main face         -> full-screen crop on it
      two people TALKING    -> stacked split-screen, left person on top
      three or more people  -> full-screen crop on whoever is talking
    A split-screen has to be earned. Both faces must be of similar size (ratio
    >= 0.65: a small face next to a big one is a doll, poster or photo inset),
    both must actually talk during the shot (a security guard or audience
    member standing behind the speaker never does), and they must be far
    enough apart that the two crops don't show the same person twice.
    Otherwise the shot goes to whoever is talking.
    """
    src_w = src_w or round(src_h * 16 / 9)
    max_zoom = max(1.0, src_h / MIN_CROP_PX)
    shots: list[Shot] = []
    for s in range(len(bounds) - 1):
        a, b = bounds[s], bounds[s + 1]
        in_shot = [t for t in tracks if t.shot == s]
        # people squeezed against the frame edge can't be framed properly
        here = [t for t in in_shot if 0.06 <= t.median(t.xs) <= 0.94] or in_shot
        if not here:
            shots.append(Shot(a, b, x=0.5, y=0.45, zoom=1.0))
            continue

        def prominence(t: FaceTrack) -> tuple[bool, float]:
            # someone who talks beats someone who only stands there; then size
            return talk_share(t, a, b) >= SPLIT_TALK_SHARE, t.median(t.hs)

        biggest = max(t.median(t.hs) for t in here)
        if biggest < WIDE_SHOT_FACE:
            # wide shot: mouths are too small to follow turn by turn, so hold
            # on one person for the whole shot
            shots.extend(single_shots(max(here, key=prominence), a, b, max_zoom))
            continue
        by_x = sorted(here, key=lambda t: t.median(t.xs))
        people = [t for t in by_x if t.median(t.hs) >= 0.45 * biggest]
        pair = [t for t in by_x if t.median(t.hs) >= 0.65 * biggest]
        if len(people) >= 3:
            for ra, rb, i in speaker_runs(people, a, b):
                shots.extend(single_shots(people[i], ra, rb, max_zoom))
            continue
        if len(pair) == 2:
            top, bottom = (face_geo(t, a, b, SPLIT_FACE_HEIGHT, max_zoom) for t in pair)
            half_w = min((src_h / g["zoom"]) * (OUT_W / (OUT_H / 2)) / src_w for g in (top, bottom))
            apart = abs(top["x"] - bottom["x"]) >= 0.6 * half_w
            both_talk = min(talk_share(t, a, b) for t in pair) >= SPLIT_TALK_SHARE
            if apart and both_talk:
                for ra, rb, i in speaker_runs(pair, a, b):
                    shots.append(Shot(ra, rb, layout="split", speaker=pair[i].person,
                                      top=dict(top), bottom=dict(bottom)))
                continue
        shots.extend(single_shots(max(here, key=prominence), a, b, max_zoom))
    return shots


def plan_with_faces(video: Path, clip: Clip, cuts: list[float], cfg: Config,
                    src_h: int, src_w: int | None = None
                    ) -> tuple[list[Shot], list[dict[str, str]]] | None:
    """Shot plan from local face analysis, or None if no faces were found."""
    found, shot_frames, bounds = analyze_faces(video, clip, cuts, cfg)
    # a face must be in at least a third of its shot's samples to count...
    tracks = [t for t in found if len(t.times) >= max(2, 0.35 * shot_frames[t.shot])]
    # ...but a shot with only a briefly seen face still beats a blind centre crop
    for s in {t.shot for t in found} - {t.shot for t in tracks}:
        brief = [t for t in found if t.shot == s and len(t.times) >= 3]
        if brief:
            tracks.append(max(brief, key=lambda t: len(t.times)))
    if not tracks:
        return None
    people = identify_people(tracks)
    return face_plan(tracks, bounds, src_h, src_w), people


def plan_shots(video: Path, clip: Clip, words: list[Word], cfg: Config, work: Path,
               index: int, src_w: int, src_h: int) -> tuple[list[Shot], list[dict[str, str]]]:
    """
    Decide the framing for a clip: local face framing first, then Gemini
    vision, then a centre crop. A framing problem never stops the run.
    """
    dur = clip.duration
    if cfg.layout == "blur":
        return [Shot(0.0, dur, layout="blur")], []
    fallback = [Shot(0.0, dur, x=0.5, y=0.45, zoom=1.0)]
    if cfg.layout == "crop" or src_w / src_h < 0.75:
        return fallback, []  # already vertical, or speaker framing switched off

    cuts = detect_cuts(video, clip)
    punches = [p - clip.start for p in clip.punch_ins if 0.5 < p - clip.start < dur - 0.5]

    if cfg.framing == "faces":
        planned = None
        try:
            planned = plan_with_faces(video, clip, cuts, cfg, src_h, src_w)
            if planned is None:
                log("no faces found in this clip")
        except ImportError:
            log("face framing needs OpenCV: .venv\\Scripts\\python -m pip install opencv-python")
        except Exception as exc:  # OpenCV or model trouble must never kill a run
            log(f"face framing failed: {type(exc).__name__}: {str(exc).strip()[:200]}")
        if planned:
            shots, people = planned
            return (apply_punches(shots, punches) if cfg.punch_ins and punches else shots), people
        if not cfg.framing_fallback_to_gemini:
            return fallback, []
        log("falling back to Gemini vision framing")

    planned_g = plan_with_gemini(video, clip, words, cfg, work, index, cuts)
    if planned_g is None:
        return fallback, []
    shots, people, vision_punches = planned_g
    for p in vision_punches:
        if all(abs(p - q) > 1.0 for q in punches):
            punches.append(p)
    if cfg.punch_ins and punches:
        shots = apply_punches(shots, sorted(punches)[:4])
    return shots, people


def plan_with_gemini(video: Path, clip: Clip, words: list[Word], cfg: Config, work: Path,
                     index: int, cuts: list[float]
                     ) -> tuple[list[Shot], list[dict[str, str]], list[float]] | None:
    """Shot plan from Gemini looking at sampled frames. None on failure."""
    dur = clip.duration
    try:
        frames = extract_frames(video, clip, cfg, work / f"frames_{index:02d}", cuts)
    except PipelineError as exc:
        log(f"frame extraction failed, using centre crop: {str(exc).splitlines()[-1]}")
        return None
    if not frames:
        return None

    rel = [Word(w.start - clip.start, w.end - clip.start, w.text)
           for w in words if clip.start <= w.start < clip.end]
    parts: list[dict] = []
    for t, data in frames:
        parts.append({"text": f"Frame at t={t:.1f}s:"})
        parts.append({"inline_data": {"mime_type": "image/jpeg",
                                      "data": base64.b64encode(data).decode("ascii")}})
    prompt = VISION_PROMPT.format(
        dur=dur,
        cuts=", ".join(f"{c:.2f}" for c in cuts) if cuts else "none (a single continuous shot)",
        transcript=transcript_as_lines(rel, window=3.0),
    )
    try:
        raw = ask_gemini_json(prompt, cfg, extra_parts=parts, temperature=0.2)
    except PipelineError as exc:
        log(f"shot planning failed, using centre crop: {str(exc).splitlines()[0]}")
        return None
    return clean_plan(raw, dur, cuts)


# ----------------------------------------------------------------------------
# timeline: dead air removed, shots cut into pieces (RULEBOOK §5 S2)
# ----------------------------------------------------------------------------


def detect_silences(video: Path, clip: Clip) -> list[tuple[float, float]]:
    """
    Stretches of the clip (seconds from clip start) where the audio really is
    quiet, measured against the clip's own loudness. Laughter, applause, music
    and singing are not silence, even though nobody is saying words.
    Returns [] if it can't be measured, which means "cut nothing".
    """
    base = ["ffmpeg", "-hide_banner", "-nostats", "-ss", f"{clip.start:.3f}",
            "-t", f"{clip.duration:.3f}", "-i", str(video.resolve()), "-vn"]

    def stderr_of(filters: str) -> str:
        try:
            proc = subprocess.run([*base, "-af", filters, "-f", "null", "-"],
                                  capture_output=True, text=True, encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return ""
        return proc.stderr or ""

    mean = re.search(r"mean_volume:\s*(-?[\d.]+) dB", stderr_of("volumedetect"))
    if not mean:
        return []
    threshold = min(-30.0, float(mean[1]) - 16.0)  # well below this clip's normal level
    log_text = stderr_of(f"silencedetect=noise={threshold:.1f}dB:d=0.35")
    starts = [float(x) for x in re.findall(r"silence_start:\s*(-?[\d.]+)", log_text)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*(-?[\d.]+)", log_text)]
    out = []
    for i, s in enumerate(starts):
        e = ends[i] if i < len(ends) else clip.duration  # silence running to the end
        if e - s >= 0.35:
            out.append((max(0.0, s), min(clip.duration, e)))
    return out


def keep_ranges(words: list[Word], clip: Clip, cfg: Config,
                silences: list[tuple[float, float]] | None = None) -> list[tuple[float, float]]:
    """
    Clip-relative ranges to keep. A pause longer than max_pause_seconds is
    shortened to max_pause_seconds by cutting its middle out.

    With `silences` (from detect_silences), a gap between words is only cut
    where the audio is also quiet, so reactions and singing survive; a silent
    tail after the last word is trimmed too. Without it, word gaps alone decide.
    """
    dur = clip.duration
    if not cfg.cut_pauses or clip.keep_pauses:
        return [(0.0, dur)]
    inside = [w for w in words if clip.start <= w.start < clip.end]
    half = cfg.max_pause_seconds / 2
    gaps: list[tuple[float, float]] = []
    for w, nxt in zip(inside, inside[1:]):
        a = est_speech_end(w) - clip.start + half
        b = nxt.start - clip.start - half
        if b - a >= 0.25:
            gaps.append((a, b))

    if silences is None:
        cuts = gaps
    else:
        cuts = []
        for a, b in gaps:
            for sa, sb in silences:
                lo, hi = max(a, sa + 0.12), min(b, sb - 0.12)
                if hi - lo >= 0.25:
                    cuts.append((lo, hi))
        if inside:  # a dead tail after the last sound
            last_sound = est_speech_end(inside[-1]) - clip.start
            for sa, sb in silences:
                lo = max(sa, last_sound) + 0.4
                if sb >= dur - 0.05 and dur - lo >= 0.25:
                    cuts.append((lo, dur))
        cuts.sort()

    keeps: list[tuple[float, float]] = []
    t = 0.0
    for a, b in cuts:
        if a > t:
            keeps.append((t, a))
        t = max(t, b)
    if t < dur:
        keeps.append((t, dur))
    return [(a, b) for a, b in keeps if b - a > 0.05] or [(0.0, dur)]


@dataclass
class Piece:
    src_start: float  # clip-relative, source timeline
    src_end: float
    out_start: float  # position in the finished clip
    shot: Shot

    @property
    def duration(self) -> float:
        return self.src_end - self.src_start


def build_pieces(keeps: list[tuple[float, float]], shots: list[Shot]) -> list[Piece]:
    pieces: list[Piece] = []
    out = 0.0
    for a, b in keeps:
        for s in shots:
            lo, hi = max(a, s.start), min(b, s.end)
            if hi - lo >= 2 / OUT_FPS:
                pieces.append(Piece(lo, hi, out, s))
                out += hi - lo
    return pieces


def to_out_time(pieces: list[Piece], t: float) -> float:
    """Map a clip-relative source time onto the finished (cut) timeline."""
    for p in pieces:
        if t < p.src_start:
            return p.out_start  # inside a cut: jump to the next kept frame
        if t <= p.src_end:
            return p.out_start + (t - p.src_start)
    return pieces[-1].out_start + pieces[-1].duration if pieces else 0.0


def piece_at(pieces: list[Piece], t_out: float) -> Piece:
    for p in pieces:
        if t_out < p.out_start + p.duration:
            return p
    return pieces[-1]


# ----------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------


def crop_box(src_w: int, src_h: int, fx: float, fy: float, zoom: float,
             aspect: float, face_frac: float) -> tuple[int, int, int, int]:
    """
    Crop rectangle (w, h, x, y) of the given aspect (w/h) around a face at
    (fx, fy), placing the face face_frac of the way down the crop.
    """
    h = src_h / max(1.0, zoom)
    w = h * aspect
    if w > src_w:
        w = src_w
        h = w / aspect
    w, h = int(w) // 2 * 2, int(h) // 2 * 2
    x = int(round(fx * src_w - w / 2))
    y = int(round(fy * src_h - face_frac * h))
    x = max(0, min(src_w - w, x))
    y = max(0, min(src_h - h, y))
    return w, h, x, y


def piece_video_chain(p: Piece, k: int, src_w: int, src_h: int, cfg: Config) -> str:
    """Filter chain for one piece: trim, frame it, normalise for concat."""
    s = p.shot
    zoom_mult = cfg.punch_zoom if s.punch else 1.0
    head = (f"[v{k}]trim=start={p.src_start:.3f}:end={p.src_end:.3f},"
            f"setpts=PTS-STARTPTS")
    tail = f"setsar=1,fps={OUT_FPS},format=yuv420p[pv{k}]"

    if s.layout == "blur":
        return (
            f"{head},split=2[bg{k}][fg{k}];"
            f"[bg{k}]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
            f"crop={OUT_W}:{OUT_H},boxblur=28:4,eq=brightness=-0.08[bgb{k}];"
            f"[fg{k}]scale={OUT_W}:-2[fgs{k}];"
            f"[bgb{k}][fgs{k}]overlay=(W-w)/2:(H-h)/2,{tail}"
        )

    max_zoom = max(1.0, src_h / MIN_CROP_PX)
    if s.layout == "split" and s.top and s.bottom:
        half_h = OUT_H // 2
        # face framing sets a zoom per person; Gemini plans share one
        boxes = [crop_box(src_w, src_h, f["x"], f["y"],
                          min(f.get("zoom") or max(1.4, s.zoom), max_zoom) * zoom_mult,
                          OUT_W / half_h, 0.42)
                 for f in (s.top, s.bottom)]
        (tw, th, tx, ty), (bw, bh, bx, by) = boxes
        return (
            f"{head},split=2[t{k}][b{k}];"
            f"[t{k}]crop={tw}:{th}:{tx}:{ty},scale={OUT_W}:{half_h}[tt{k}];"
            f"[b{k}]crop={bw}:{bh}:{bx}:{by},scale={OUT_W}:{half_h}[bb{k}];"
            f"[tt{k}][bb{k}]vstack=inputs=2,{tail}"
        )

    w, h, x, y = crop_box(src_w, src_h, s.x, s.y, min(s.zoom, max_zoom) * zoom_mult,
                          OUT_W / OUT_H, 0.36)
    return f"{head},crop={w}:{h}:{x}:{y},scale={OUT_W}:{OUT_H},{tail}"


def build_filter(pieces: list[Piece], src_w: int, src_h: int, has_audio: bool,
                 cfg: Config, subtitle_name: str | None) -> str:
    n = len(pieces)
    parts: list[str] = []
    parts.append(f"[0:v]split={n}" + "".join(f"[v{k}]" for k in range(n)) if n > 1
                 else "[0:v]null[v0]")
    if has_audio:
        parts.append(f"[0:a]asplit={n}" + "".join(f"[a{k}]" for k in range(n)) if n > 1
                     else "[0:a]anull[a0]")

    for k, p in enumerate(pieces):
        parts.append(piece_video_chain(p, k, src_w, src_h, cfg))
        if has_audio:
            fade = min(0.012, p.duration / 4)
            parts.append(
                f"[a{k}]atrim=start={p.src_start:.3f}:end={p.src_end:.3f},asetpts=PTS-STARTPTS,"
                f"aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
                f"afade=t=in:d={fade:.3f},afade=t=out:st={max(0.0, p.duration - fade):.3f}:d={fade:.3f}"
                f"[pa{k}]"
            )

    inputs = "".join(f"[pv{k}]" + (f"[pa{k}]" if has_audio else "") for k in range(n))
    parts.append(f"{inputs}concat=n={n}:v=1:a={1 if has_audio else 0}[cv]"
                 + ("[aout]" if has_audio else ""))

    video = "[cv]"
    post: list[str] = []
    if cfg.color_grade:
        post.append("eq=contrast=1.06:saturation=1.15:brightness=0.01,unsharp=5:5:0.5:5:5:0")
    if subtitle_name:
        # Relative filename only. ffmpeg runs with cwd set to the work dir, which
        # sidesteps the Windows "C:\path" colon-escaping problem in filtergraphs.
        # ass rather than subtitles: only ass exposes shaping=complex, which
        # non-Latin scripts need to render correctly.
        post.append(f"ass={subtitle_name}:shaping=complex")
    parts.append(f"{video}{','.join(post) if post else 'null'}[vout]")
    return ";".join(parts)


def ass_color(hex_color: str, alpha: int = 0) -> str:
    """#RRGGBB -> ASS &HAABBGGRR"""
    h = hex_color.strip().lstrip("#").upper()
    if not re.fullmatch(r"[0-9A-F]{6}", h):
        h = "FFFFFF"
    return f"&H{alpha:02X}{h[4:6]}{h[2:4]}{h[0:2]}"


def ass_inline_color(hex_color: str) -> str:
    return "\\c" + ass_color(hex_color)[:2] + ass_color(hex_color)[4:] + "&"


ASS_HEADER = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{font},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,8,4,5,40,40,0,1
Style: Hook,{hook_font},58,&H00000000,&H00000000,&H00FFFFFF,&H00FFFFFF,-1,0,0,0,100,100,0,0,3,18,0,8,80,80,0,1
Style: Mark,Arial Black,40,&H66FFFFFF,&H66FFFFFF,&H99000000,&H00000000,-1,0,0,0,100,100,1,0,1,2,0,5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def wrap_hook(text: str, width: int = 24, max_lines: int = 2) -> str:
    """
    The context box as ASS text: at most two lines of `width` characters, so it
    never runs off the sides of the frame. Anything longer is left out; a
    context line that needs three lines is not a hook.
    """
    text = strip_emoji(text.strip().replace("{", "(").replace("}", ")").replace("\\", ""))
    lines: list[str] = []
    for word in text.split():
        if lines and len(lines[-1]) + 1 + len(word) <= width:
            lines[-1] += " " + word
        else:
            lines.append(word)
    if not lines or len(lines) > max_lines or any(len(l) > width + 4 for l in lines):
        return ""
    return "\\N".join(lines)


def caption_chunks(words: list[tuple[Word, str]], cfg: Config,
                   speaker_of: Callable[[float], str]) -> list[list[tuple[Word, str]]]:
    """
    Group (word, display_text) pairs into on-screen chunks: at most
    caption_words_per_line words and caption_max_chars characters, never
    spanning a speaker change, a sentence end or a pause.
    """
    chunks: list[list[tuple[Word, str]]] = []
    cur: list[tuple[Word, str]] = []
    for w, txt in words:
        if cur:
            joined = len(" ".join(t for _, t in cur)) + 1 + len(txt)
            prev_w, prev_t = cur[-1]
            if (len(cur) >= max(1, cfg.caption_words_per_line)
                    or joined > cfg.caption_max_chars
                    or w.start - prev_w.end > 0.6
                    or prev_w.text.rstrip().endswith(SENTENCE_END)
                    or speaker_of(w.start) != speaker_of(prev_w.start)):
                chunks.append(cur)
                cur = []
        cur.append((w, txt))
    if cur:
        chunks.append(cur)
    return chunks


def build_caption_file(words: list[Word], clip: Clip, pieces: list[Piece],
                       cfg: Config, dest: Path) -> bool:
    """Write the .ass overlay for a clip, timed to the finished (cut) timeline."""
    total = pieces[-1].out_start + pieces[-1].duration
    bad = [b.lower() for b in (cfg.censor_words or DEFAULT_CENSOR)]

    speaker_ids = [p["id"] for p in clip.people] or []

    def speaker_of(t_out: float) -> str:
        return piece_at(pieces, t_out).shot.speaker

    def color_for(t_out: float) -> str:
        sid = speaker_of(t_out)
        if sid and sid not in speaker_ids:
            speaker_ids.append(sid)
        idx = speaker_ids.index(sid) if sid in speaker_ids else 0
        palette = cfg.speaker_colors or ["#FFFFFF"]
        return palette[idx % len(palette)]

    keywords = {
        norm_token(t) for k in clip.keywords for t in k.split()
        if len(norm_token(t)) >= 3 and norm_token(t) not in KEYWORD_STOPWORDS
    }

    events: list[str] = []
    if cfg.burn_captions:
        timed: list[tuple[Word, str]] = []
        for w in words:
            if w.end <= clip.start or w.start >= clip.end:
                continue
            txt = display_text(w, bad)
            if not txt:
                continue
            a = to_out_time(pieces, max(0.0, w.start - clip.start))
            b = to_out_time(pieces, min(clip.duration, w.end - clip.start))
            timed.append((Word(a, max(b, a + 0.12), w.text), txt.upper()))

        for chunk in caption_chunks(timed, cfg, speaker_of):
            start = chunk[0][0].start
            end = min(total, max(chunk[-1][0].end, start + 0.3))
            if end <= start:
                continue
            layout = piece_at(pieces, start).shot.layout
            y = OUT_H // 2 if layout == "split" else int(cfg.caption_vertical_pos * OUT_H)
            base = color_for(start)
            text_parts = []
            for _, txt in chunk:
                if norm_token(txt) in keywords:
                    text_parts.append("{" + ass_inline_color(cfg.keyword_color)
                                      + "\\fscx112\\fscy112}" + txt
                                      + "{" + ass_inline_color(base) + "\\fscx100\\fscy100}")
                else:
                    text_parts.append(txt)
            length = len(" ".join(t for _, t in chunk))
            fit = min(100, int(100 * cfg.caption_max_chars / max(1, length)))
            # small pop-in so captions feel animated rather than static
            effect = (f"{{\\an5\\pos({OUT_W // 2},{y}){ass_inline_color(base)}"
                      f"\\fad(40,40)\\fscx{int(fit * 0.9)}\\fscy{int(fit * 0.9)}"
                      f"\\t(0,90,\\fscx{fit}\\fscy{fit})}}")
            events.append(f"Dialogue: 1,{ass_time(start)},{ass_time(end)},Cap,,0,0,0,,"
                          f"{effect}{' '.join(text_parts)}")

    hook = wrap_hook(clip.hook_text) if cfg.hook_text else ""
    if hook:
        events.append(f"Dialogue: 2,{ass_time(0)},{ass_time(min(total, cfg.hook_seconds))},"
                      f"Hook,,0,0,0,,{{\\an8\\pos({OUT_W // 2},240)\\fad(0,150)}}{hook}")

    if cfg.watermark.strip():
        mark = strip_emoji(cfg.watermark.strip()).replace("{", "(").replace("}", ")")
        events.append(f"Dialogue: 0,{ass_time(0)},{ass_time(total)},Mark,,0,0,0,,"
                      f"{{\\an5\\pos({OUT_W // 2},{int(OUT_H * 0.76)})}}{mark.upper()}")

    if not events:
        return False
    header = ASS_HEADER.format(font=cfg.caption_font, size=cfg.caption_font_size,
                               hook_font=cfg.hook_font)
    dest.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return True


X264_ARGS = ["-c:v", "libx264", "-preset", "medium", "-crf", "20",
             "-profile:v", "high", "-pix_fmt", "yuv420p"]
NVENC_ARGS = ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr",
              "-cq", "21", "-b:v", "0", "-profile:v", "high", "-pix_fmt", "yuv420p"]
_encoder: list[str] | None = None  # decided once per run


def encoder_args(cfg: Config) -> list[str]:
    """NVENC (GPU) when it actually works on this machine, otherwise x264 (CPU)."""
    global _encoder
    if cfg.video_encoder == "x264":
        return X264_ARGS
    if _encoder is None:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
             "-i", f"color=black:s={OUT_W}x{OUT_H}:d=0.2", *NVENC_ARGS, "-f", "null", "-"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if proc.returncode == 0:
            _encoder = NVENC_ARGS
            log("video encoder: NVENC (GPU)")
        else:
            lines = (proc.stderr or "").strip().splitlines()
            why = next((l for l in lines if "driver" in l.lower()), lines[-1] if lines else "")
            log(f"video encoder: x264 (CPU); NVENC unavailable: {why.split('] ')[-1].strip()}")
            _encoder = X264_ARGS
    return _encoder


def render_clip(video: Path, clip: Clip, words: list[Word], caption_words: list[Word],
                cfg: Config, work: Path, out_dir: Path, index: int,
                src: tuple[int, int, bool]) -> Path:
    src_w, src_h, has_audio = src

    shots, people = plan_shots(video, clip, words, cfg, work, index, src_w, src_h)
    clip.people = people
    silences = (detect_silences(video, clip)
                if has_audio and cfg.cut_pauses and not clip.keep_pauses else [])
    keeps = keep_ranges(words, clip, cfg, silences)
    pieces = build_pieces(keeps, shots)
    if not pieces:
        raise PipelineError(f"clip '{clip.title}' has nothing left to render")
    clip.shots = [asdict(s) for s in shots]
    clip.kept_ranges = [[round(a, 2), round(b, 2)] for a, b in keeps]
    clip.final_seconds = round(pieces[-1].out_start + pieces[-1].duration, 2)

    sub_name = None
    sub_path = work / f"caps_{index:02d}.ass"
    if build_caption_file(caption_words, clip, pieces, cfg, sub_path):
        sub_name = sub_path.name

    filename = f"{index:02d}_{safe_filename(clip.title)}.mp4"
    out_path = out_dir / filename

    graph = build_filter(pieces, src_w, src_h, has_audio, cfg, sub_name)
    graph_file = work / f"graph_{index:02d}.txt"
    graph_file.write_text(graph, encoding="utf-8")

    def command(enc: list[str]) -> list[str]:
        return [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{clip.start:.3f}",
            "-t", f"{clip.duration:.3f}",
            "-i", str(video.resolve()),
            # read the graph from a file: dozens of pieces overflow the Windows
            # command-line limit otherwise
            "-/filter_complex", graph_file.name,
            "-map", "[vout]",
            *(["-map", "[aout]"] if has_audio else []),
            *enc,
            "-r", str(OUT_FPS), "-g", str(OUT_FPS * 2),
            *(["-c:a", "aac", "-b:a", "160k", "-ar", "48000"] if has_audio else []),
            "-movflags", "+faststart",
            str(out_path.resolve()),
        ]

    enc = encoder_args(cfg)
    try:
        run(command(enc), cwd=work)
    except PipelineError:
        if enc is not NVENC_ARGS:
            raise
        global _encoder
        _encoder = X264_ARGS
        log("NVENC failed on this clip, re-rendering on the CPU")
        run(command(X264_ARGS), cwd=work)
    return out_path


# ----------------------------------------------------------------------------
# post kit, history and posting schedule (RULEBOOK §8, §9, §11, §13)
# ----------------------------------------------------------------------------


def load_history(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("clips"), list):
                return data
        except json.JSONDecodeError:
            log(f"warning: {path.name} is unreadable, starting a new one")
    return {"clips": []}


def already_rendered(clip: Clip, video_id: str, history: dict[str, Any]) -> bool:
    """True if at least half of this clip was rendered from this video before."""
    for h in history["clips"]:
        if h.get("video_id") != video_id:
            continue
        overlap = min(clip.end, h["end"]) - max(clip.start, h["start"])
        if overlap > 0.5 * min(clip.duration, h["end"] - h["start"]):
            return True
    return False


def next_post_slots(history: dict[str, Any], post_times: list[str], n: int,
                    now: datetime | None = None) -> list[datetime]:
    """
    Spread clips over the configured daily posting times, continuing after the
    last slot already handed out, so a batch never goes up all at once.
    """
    now = now or datetime.now()
    times: list[tuple[int, int]] = []
    for t in post_times or ["12:30", "17:30", "21:00"]:
        m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", str(t))
        if m and int(m[1]) < 24 and int(m[2]) < 60:
            times.append((int(m[1]), int(m[2])))
    times = sorted(set(times)) or [(12, 30), (17, 30), (21, 0)]

    after = now + timedelta(minutes=30)
    for h in history["clips"]:
        try:
            after = max(after, datetime.fromisoformat(h.get("post_at", "")) + timedelta(minutes=1))
        except ValueError:
            pass

    slots: list[datetime] = []
    day = after.date()
    while len(slots) < n:
        for hh, mm in times:
            slot = datetime(day.year, day.month, day.day, hh, mm)
            if slot >= after and len(slots) < n:
                slots.append(slot)
        day += timedelta(days=1)
    return slots


def post_kit(clip: Clip, meta: dict[str, Any]) -> str:
    """Everything needed to publish one clip by hand, in paste order."""
    credit_name = meta.get("channel") or meta.get("uploader") or "the original creator"
    credit_url = meta.get("webpage_url") or ""
    gate = ", ".join(k for k, v in clip.gate.items() if v) or "none"
    lines = [
        "TITLE",
        clip.title,
        "",
        "DESCRIPTION",
        clip.caption.strip(),
        "",
        f"Credit: {credit_name}" + (f" - full video: {credit_url}" if credit_url else ""),
        "",
        " ".join(clip.hashtags),
        "",
        "TAGS",
        ", ".join(clip.tags),
        "",
        "PINNED COMMENT",
        clip.pinned_comment,
        "",
        "SETTINGS",
        "Category: Entertainment",
        f"Suggested post time: {clip.post_at.replace('T', ' ')}",
        "Optional in the app: a trending sound at low volume under the dialogue.",
        "",
        "WHY THIS CLIP",
        f"score {clip.score} | rulebook gate {clip.gate_points}/10 ({gate}) | "
        f"trigger: {clip.trigger}",
        clip.reason,
        f"source {clip.start:.1f}s-{clip.end:.1f}s, {clip.final_seconds:.1f}s after cuts",
    ]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------


def probe_duration(video: Path) -> float:
    out = run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video),
    ])
    try:
        return float(out.strip())
    except ValueError:
        return 0.0


def folder_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0


def remove_other_downloads(work: Path, video_id: str) -> None:
    """
    Delete source downloads (finished or partial) of any other video, left
    behind by an earlier run that failed. The current video's download is kept
    so a retry can resume or reuse it.
    """
    mine = f"source_{safe_filename(video_id, 40)}."
    freed = 0
    for f in work.glob("source*"):
        if f.is_file() and not f.name.startswith(mine):
            freed += f.stat().st_size
            f.unlink(missing_ok=True)
    if freed:
        log(f"deleted leftover downloads of earlier videos ({freed / 1e6:.0f} MB)")


def process(url: str, cfg: Config) -> None:
    require_tools()

    out_root = Path(cfg.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    work = Path(cfg.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    history_path = out_root / "history.json"
    history = load_history(history_path)

    step("Fetching video info")
    meta = fetch_metadata(url)
    title = meta.get("title", "untitled")
    video_id = str(meta.get("id") or safe_filename(title))
    duration = float(meta.get("duration") or 0)
    trending = is_trending(meta)
    log(f"{title}  ({duration / 60:.1f} min){'  [trending: +1 gate point]' if trending else ''}")
    remove_other_downloads(work, video_id)

    step("Looking for captions")
    words, lang = fetch_youtube_captions(url, work, meta, cfg.subtitle_langs)
    if words:
        log(f"found YouTube captions: {len(words)} words")
    else:
        log("no caption track available, will transcribe locally after download")

    step("Downloading source video")
    # Named by video ID so a cached download is only ever reused for the same
    # video. A fixed name once cut one video's transcript onto another's footage.
    video_path = work / f"source_{safe_filename(video_id, 40)}.mp4"
    if video_path.exists():
        cached = probe_duration(video_path)
        if duration and abs(cached - duration) > 5:
            log(f"cached download is {cached / 60:.1f} min but the video is "
                f"{duration / 60:.1f} min; downloading again")
            video_path.unlink()
        else:
            log("already downloaded, reusing")
    if not video_path.exists():
        run([
            *YTDLP,
            "-f", (
                f"bv*[height<={cfg.max_source_height}][ext=mp4]+ba[ext=m4a]/"
                f"b[height<={cfg.max_source_height}][ext=mp4]/b"
            ),
            "--merge-output-format", "mp4",
            "--no-warnings",
            # Flaky DNS or wifi: keep trying with growing waits, and resume the
            # partial file, instead of giving up after a few seconds.
            "--retries", "30", "--fragment-retries", "30", "--retry-sleep", "exp=1:30",
            "-o", str(video_path),
            url,
        ])
    log(f"{video_path.stat().st_size / 1_000_000:.0f} MB")

    if not words:
        step("Transcribing")
        words, lang = transcribe_locally(video_path, cfg)
        log(f"{len(words)} words")

    if not words:
        raise PipelineError("Could not obtain a transcript by any method.")

    if duration <= 0:
        duration = probe_duration(video_path)
    src = probe_video(video_path)

    step("Asking Gemini to pick the best moments")
    lines = transcript_sentences(words)
    log(f"transcript split into {len(lines)} sentences")
    candidates = select_clips_with_gemini(lines, meta, cfg, trending)
    log(f"{len(candidates)} candidates returned")

    candidates, dropped = sanitize_clips(candidates, cfg, duration, words, lines)

    def passes(c: Clip) -> bool:
        return c.score >= cfg.score_threshold and c.gate_points >= cfg.min_gate_points

    approved = [c for c in candidates if passes(c)]
    rejected = [c for c in candidates if not passes(c)]

    if not cfg.allow_repeats:
        repeats = [c for c in approved if already_rendered(c, video_id, history)]
        for c in repeats:
            c.reason = "[already rendered from this video; duplicate upload risk] " + c.reason
        approved = [c for c in approved if c not in repeats]
        rejected = repeats + rejected
    for c in approved[cfg.max_clips_to_render:]:
        c.reason = f"[passed, but over max_clips_to_render={cfg.max_clips_to_render}] " + c.reason
    rejected = approved[cfg.max_clips_to_render:] + rejected + dropped
    approved = approved[:cfg.max_clips_to_render]

    step(f"Quality gate (score >= {cfg.score_threshold}, rulebook gate >= {cfg.min_gate_points}/10)")
    for c in approved:
        log(f"KEEP  {c.score:3d}  gate {c.gate_points:2d}  {c.duration:5.1f}s  {c.title}")
    for c in rejected:
        log(f"drop  {c.score:3d}  gate {c.gate_points:2d}  {c.duration:5.1f}s  {c.title}  -- {c.reason}")

    if not approved:
        if candidates:
            hint = ("Every candidate was below the score or rulebook gate. Run again with a\n"
                    "lower bar (e.g. --threshold 65), or pick a source with stronger moments.")
        else:
            hint = ("No candidate survived the length and overlap checks (reasons above), so a\n"
                    "lower --threshold won't help. Run again: selection varies per run, and\n"
                    "the Flash models (sharper than Flash-Lite) are back after the daily reset.")
        print(f"\nNothing to render. {hint}\nThe download is kept in {work.name}, "
              "so a retry on this video skips it.")
        return

    caption_words = words
    if cfg.burn_captions and needs_translation(lang, cfg.caption_language):
        step(f"Translating captions ({lang} -> {cfg.caption_language})")
        caption_words = translate_captions(approved, words, title, cfg)
        log(f"{len(caption_words)} translated words")

    out_dir = out_root / f"{datetime.now():%Y%m%d}_{safe_filename(video_id, 20)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    step(f"Rendering {len(approved)} clips")
    # Only clips that still exist on disk hold a posting slot. Runs whose
    # folders were deleted must not push new clips a week into the future.
    waiting = {"clips": [h for h in history["clips"]
                         if (out_root / str(h.get("folder", "")) / str(h.get("file", ""))).is_file()]}
    slots = next_post_slots(waiting, cfg.post_times, len(approved))
    for i, (clip, slot) in enumerate(zip(approved, slots), start=1):
        log(f"[{i}/{len(approved)}] {clip.title}")
        clip.post_at = slot.isoformat(timespec="minutes")
        path = render_clip(video_path, clip, words, caption_words, cfg, work, out_dir, i, src)
        clip.file = path.name
        path.with_suffix(".txt").write_text(post_kit(clip, meta), encoding="utf-8")
        layouts = sorted({s["layout"] + ("+punch" if s["punch"] else "") for s in clip.shots})
        log(f"    {clip.final_seconds:.1f}s, {len(clip.shots)} shots ({', '.join(layouts)}), "
            f"post {clip.post_at.replace('T', ' ')}")
        history["clips"].append({
            "video_id": video_id, "start": clip.start, "end": clip.end,
            "title": clip.title, "folder": out_dir.name, "file": clip.file,
            "post_at": clip.post_at,
        })
        history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "source_url": url,
        "source_title": title,
        "source_channel": meta.get("channel") or meta.get("uploader"),
        "threshold": cfg.score_threshold,
        "min_gate_points": cfg.min_gate_points,
        "approved": [asdict(c) for c in approved],
        "rejected": [asdict(c) for c in rejected],
    }
    (out_dir / "clips.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if cfg.keep_work_files:
        log(f"kept the download and temp files in {work} (--keep-work)")
    else:
        freed = folder_size(work)
        shutil.rmtree(work, ignore_errors=True)
        log(f"cleaned up: deleted the downloaded video and temp files ({freed / 1e6:.0f} MB freed)")

    print(f"\nDone. {len(approved)} clips in {out_dir}")
    print("Each clip has a .txt post kit next to it: title, description, hashtags,")
    print("pinned comment and a suggested post time. Post one per slot, not all at once.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Turn a YouTube video into short clips, locally.")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("-c", "--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--threshold", type=int, help="override the score threshold")
    parser.add_argument("--focus", help="override channel_focus (the person the channel is about)")
    parser.add_argument("--keep-work", action="store_true",
                        help="keep the downloaded source and temp files even after a successful run")
    parser.add_argument("--allow-repeats", action="store_true",
                        help="render moments even if they were rendered from this video before")
    args = parser.parse_args()

    # Titles and transcripts are often non-Latin; don't crash when output is
    # redirected to a file or pipe using the legacy Windows code page.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    try:
        cfg = Config.load(Path(args.config))
        if args.threshold is not None:
            cfg.score_threshold = args.threshold
        if args.focus is not None:
            cfg.channel_focus = args.focus
        if args.keep_work:
            cfg.keep_work_files = True
        if args.allow_repeats:
            cfg.allow_repeats = True
        process(args.url, cfg)
    except PipelineError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        print("Any download is kept, so running the same command again skips it.",
              file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Any download is kept for the next run.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
