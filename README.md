# Local YouTube clipper

Takes a long YouTube video, picks the strongest moments, and renders them as
vertical clips with burned-in captions into a folder on your PC.

Nothing is uploaded anywhere. You review the clips and post them yourself.

**Running cost: $0.** Gemini's free tier covers the clip selection; everything
else runs locally.

---

## One-time setup (Windows)

### 1. Python

Install Python 3.10 or newer from [python.org](https://www.python.org/downloads/).
Tick **"Add Python to PATH"** on the first screen of the installer.

Check it worked, in PowerShell:

```powershell
python --version
```

### 2. ffmpeg

The easiest route is winget:

```powershell
winget install Gyan.FFmpeg
```

Close and reopen PowerShell, then confirm:

```powershell
ffmpeg -version
```

If winget isn't available, download a build from
[gyan.dev](https://www.gyan.dev/ffmpeg/builds/), unzip it, and add its `bin`
folder to your PATH manually.

### 3. Python packages

From this folder, create a private environment for this project and install
into it:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

`.venv` only works on the machine that created it, so it isn't included here.
Create your own with the commands above.

**Downloaded on your computer, not included in this repo:**

- `models/`: the two small face-detection models (YuNet and SFace from the
  OpenCV model zoo). They download automatically the first time you run the
  clipper.
- ffmpeg (step 2) and your Gemini API key (step 4).
- Optional: the Whisper speech model, only if you install `faster-whisper`
  (see "No captions found" below). It also downloads on first use.

`clips/` and `.clipper-work/` are created when you run it.

### 4. Gemini API key

Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

Set it permanently:

```powershell
setx GEMINI_API_KEY "paste-your-key-here"
```

Then **close and reopen PowerShell** — `setx` only affects new sessions.

Verify:

```powershell
echo $env:GEMINI_API_KEY
```

---

## Running it

From this folder:

```powershell
.venv\Scripts\python main.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

That's the whole interface. Every step follows `RULEBOOK.md`, the standard
built from studying faceless clip channels that get 10M-300M views per Short.
It will:

1. Fetch the video's title and caption track, then download the video once
2. Ask Gemini for candidate moments, scored on attention *and* on the
   rulebook's 10-point gate (understandable muted in 3s, emotion peak, second
   famous person, stakes, ends on the payoff, broad appeal, fits your channel)
3. Drop anything that fails either gate, and anything already rendered before
4. Build every clip from whole sentences. It opens on the sentence that sets
   the moment up (skipping "so", "um" and the like) and ends after the
   punchline and the laugh or reaction that follows, never in mid-sentence.
   Each clip is found by matching the model's quoted lines against the
   transcript, so a wrong line number can't put it in the wrong place.
5. Plan the framing on your PC with face detection. It finds the exact camera
   cuts, then:
   - a single speaker gets a full-screen crop
   - two people get a stacked split-screen
   - in a group, the crop follows whoever's mouth is moving
   - wide shots hold on the most prominent person
   - punchlines get a quick zoom

   Each person is recognised across cuts and keeps their caption colour. No
   blurred letterbox, and no API calls for this step.
6. Render 1080x1920 with true silences cut out (laughter, applause and singing
   are kept), a light colour grade, and 1-3 word captions. Captions are coloured per speaker, keywords are yellow, sound tags
   are removed and profanity is starred. Optionally adds a context box at the
   top and your watermark.
7. Put each run in its own folder `clips/<date>_<video id>/`, with a `.txt`
   **post kit** next to every clip containing the title, description with
   credit, hashtags, tags, pinned comment and a suggested post time

Useful flags:

```powershell
# try a looser score gate without re-running everything from scratch
python main.py "URL" --threshold 65

# say who the channel is about (overrides channel_focus in config.yaml)
python main.py "URL" --focus "Ashneer Grover"

# keep the downloaded video even after a successful run
python main.py "URL" --keep-work

# render a moment again even though it was rendered from this video before
python main.py "URL" --allow-repeats
```

### Disk space and the download cache

The source video (often 1-2 GB) is managed for you:

- **Successful run:** the download and all temp files are deleted, and the
  log says how much space was freed.
- **Failed run**, or nothing passed the gate: the download is kept, so running
  the same command again skips it.
- **Starting a different video:** leftover downloads of other videos are
  deleted first. Downloads are named by video ID, so one video's footage is
  never reused for another.

### When Gemini is busy or out of quota

Every model has its own free daily quota. The clipper tries
`gemini_model`, then each of `gemini_fallback_models` in order: Flash, then
the Flash-Lite models. A model that reports its daily quota is used up is
skipped for the rest of the run. Busy (503) and per-minute limits are retried
after 10s, 30s and 60s. You only get an error when every model is out for the
day; quotas reset at midnight Pacific time.

### Posting the clips

Open the `.txt` next to each clip and paste it in order: title, description
(credit included), hashtags, tags, then pin the comment. **Post one clip per
suggested time slot.** The slots come from `post_times` in `config.yaml` and
pick up after the last slot already handed out, across runs. Uploading a whole
batch at once, or the same clip twice, is what gets a new channel's Shorts
shown to nobody.

`clips/history.json` remembers every clip rendered. Don't delete it unless you
want repeats.

---

## Tuning it

Everything lives in `config.yaml`, and each setting says which rulebook
section it comes from. The ones that matter most:

- **`channel_focus`**: the one person (or duo/show) your channel is about.
  Channels built on one personality win, and channels that clip "everyone" don't.
- **`watermark`**: your `@handle`, burned in semi-transparently.
- **`score_threshold`** and **`min_gate_points`**: the two quality gates
  (75 and 7/10 to start). Every candidate's score, gate answers and reasoning
  are in `clips.json`, rejects included, so calibrate against real output.
- **`layout`**: `speaker` (default) crops to whoever is talking and uses a
  split-screen for conversations. `crop` is a fixed centre crop. `blur` is the
  old letterbox look, which top channels don't use.
- **`framing`**: `faces` (default) finds people with local face detection,
  which is precise, fast and uses no API quota. `gemini` asks Gemini to look
  at frames instead (1 extra API call per clip). If `faces` can't find anyone,
  it falls back to Gemini, then to a centre crop.
- **`video_encoder`**: `auto` uses your NVIDIA GPU (NVENC) when it works,
  otherwise the CPU. The run log says which one it used and why.
- **`caption_language`** and **`title_language`**: keep them the same. For a
  Hindi video either translate (`English`) or keep Hindi (`""` plus
  `caption_font: Nirmala UI`, since Impact has no Devanagari).
- **`cut_pauses`** / **`max_pause_seconds`**: dead air longer than 0.5s is
  shortened to 0.5s. Gemini keeps a pause when the silence is the joke.
- **`post_times`**: the daily slots used for suggested post times.

The rulebook also says to pick sources you're allowed to clip. Many creators
run clipping programs; check before building a channel on someone. Shows that
were taken down, or whose owners claim every clip, will get your Shorts
blocked or shown to nobody.

---

## When something goes wrong

**`'yt-dlp' is not recognized`** — the `pip install` put it somewhere not on
your PATH. Use `python -m yt_dlp` to confirm it's installed, then either add
Python's `Scripts` folder to PATH or reinstall Python with the PATH box ticked.

**Captions don't appear in the output** — the font name in `config.yaml` has to
match an installed font exactly. `Arial Black` and `Impact` are safe on Windows.

**"No captions found"** — the video has no auto-caption track. Install the
fallback transcriber:

```powershell
pip install faster-whisper
```

The first run downloads the model (~150 MB for `base`), then transcribes on CPU.
Budget roughly 10–20 minutes for a 60-minute video.

**"daily free quota used up" for every model** — the free tier is spent for
today on all models in `config.yaml`. It resets at midnight Pacific time. Any
download is kept, so the retry is quick.

**"busy (503)" or "rate limited (429)"** — temporary. The clipper already
retries and falls back to the next model; if everything is busy, wait a few
minutes and run the same command again.

**A clip starts without context or ends too early** — clips are cut on whole
sentences, so this means the model chose the wrong first or last sentence.
Open `clips.json`: `start_line`, `end_line`, `hook` and `reason` show what it
picked. Running again gives a fresh selection, and the Flash models choose
better than Flash-Lite, which is only used when the Flash quota is spent.

**Captions are in the wrong language or look like nonsense** — check the
"using auto caption track" line at the start of the run. It should name the
language spoken in the video (for example `hi-orig` for Hindi).

**Download fails or stalls** — YouTube changes things often and `yt-dlp` ships
fixes fast. Update it first:

```powershell
pip install -U yt-dlp
```

---

## What this does not do

- No smooth camera *panning*. The crop is fixed for each camera shot and
  re-aimed about every 2s if someone moves around, so it jumps rather than
  glides. Check `shots` in `clips.json` if a clip looks off.
- Speaker detection reads mouth movement. It is reliable on close-ups and
  two-shots. In wide group shots the faces are too small, so it holds on the
  most prominent person instead of guessing.
- No B-roll, no music, no sound effects. Add a trending sound in the app when
  posting if you want one.
- No uploading. By design — see the note below.

## A note on posting

Uploading these by hand is not just the cheap option. You see every clip before
it goes out, you can add trending audio (which no API can do), and native app
uploads tend to do better in the feed than API-published posts. For ten clips a
week that's a couple of minutes a day.
