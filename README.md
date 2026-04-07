# Local Transcript Studio

A desktop application for transcribing local audio and video files entirely offline using [faster-whisper](https://github.com/SYSTRAN/faster-whisper). No data leaves your machine.

## Features

- Transcribes MP4, MKV, MOV, AVI, WebM, MP3, WAV, M4A, and FLAC files
- Outputs a plain-text transcript (`_transcript.txt`) and a timed subtitle file (`_subtitles.srt`) alongside the source file
- **Multi-speaker diarization** — labels each segment with who is speaking (`SPEAKER_00`, `SPEAKER_01`, …) using [pyannote.audio](https://github.com/pyannote/pyannote-audio) _(optional)_
- GPU acceleration via CUDA with automatic fallback to CPU
- Real-time progress bar showing percentage, elapsed time, ETA, and transcription speed
- Stop button to cancel an in-progress transcription
- Drag-and-drop file input (requires `tkinterdnd2`)
- YouTube URL input — downloads audio and transcribes it (requires `yt-dlp` and FFmpeg)
- AI-powered meeting summarisation via any OpenAI-compatible API
- Per-session log files written to `~/LocalTranscriptLogs/`

## Requirements

- Python 3.10+
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [tkinterdnd2](https://github.com/pmgagne/tkinterdnd2) _(optional — enables drag and drop)_
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) _(optional — enables YouTube URL input)_
- [FFmpeg](https://ffmpeg.org/) with `ffprobe` on `PATH` _(optional — enables ETA calculation and YouTube audio extraction)_
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) _(optional — enables speaker diarization)_

For GPU acceleration, a CUDA-capable GPU with cuDNN 9 installed is required.

## Installation

Clone the repo and install using `pip`:

```bash
# Core only (transcription)
pip install .

### FFmpeg (optional)

FFmpeg provides two things: `ffprobe` for ETA calculation during transcription, and the audio extraction pipeline used when downloading YouTube videos. It is also required for diarization of video files (MP4, MKV, etc.).

**Windows**

1. Download a build from [ffmpeg.org/download](https://ffmpeg.org/download.html) (e.g. the *gyan.dev* release) or install via winget:
   ```
   winget install ffmpeg
   ```
2. Extract the archive and add the `bin\` folder to your `PATH`, or place `ffmpeg.exe` and `ffprobe.exe` somewhere already on `PATH` (e.g. `C:\Windows\System32`).
3. Verify with `ffprobe -version` in a new terminal.

**macOS**

```bash
brew install ffmpeg
```

**Linux**

```bash
# Debian / Ubuntu
sudo apt install ffmpeg

# Fedora
sudo dnf install ffmpeg
```

## Usage

```bash
python main.py
# or, if installed via pip:
local-transcript
```

1. Drag a media file onto the drop zone, or click **Browse MP4 / Media**
2. Select a model (`medium` or `large-v3`) and preferred device (`cuda` or `cpu`)
3. Click **Start Transcription**
4. Output files are written to the same folder as the source file

To cancel a running transcription, click **Stop**. Any segments already processed are discarded and no output files are written.

## Speaker Diarization

Speaker diarization identifies who is speaking in each segment of the transcript. When enabled, output files are annotated with speaker labels:

**`_transcript.txt`**
```
SPEAKER_00: Hello, welcome to the meeting.
SPEAKER_01: Thanks for having me.
SPEAKER_00: Let's get started.
```

**`_subtitles.srt`**
```
1
00:00:00,000 --> 00:00:03,200
[SPEAKER_00] Hello, welcome to the meeting.

2
00:00:03,500 --> 00:00:06,100
[SPEAKER_01] Thanks for having me.
```

### Setup

1. Install the package:
   ```bash
   pip install pyannote-audio
   ```

2. Create a free account at [huggingface.co](https://huggingface.co), then accept the model terms at:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

3. Generate a read token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

4. In the app, open **Settings → Speaker Diarization**, paste your token, check **Enable speaker diarization**, and click **Save Settings**.

The model is downloaded on first use and cached locally. All subsequent inference runs fully offline — the token is only needed for the initial download.

> **Note:** Diarization runs after transcription completes and adds extra processing time proportional to the audio length.

## Models

| Model | Speed | Quality |
|-------|-------|---------|
| `medium` | Faster | Good |
| `large-v3` | Slower | Best |

Models are downloaded automatically by faster-whisper on first use and cached locally.

## Output files

Given an input file `interview.mp4`, two files are created in the same directory:

- `interview_transcript.txt` — plain text, one paragraph per segment (prefixed with speaker labels when diarization is enabled)
- `interview_subtitles.srt` — standard SRT format with timestamps (prefixed with speaker labels when diarization is enabled)

## Project structure

```
src/
├── constants.py      # Shared constants
├── logging_setup.py  # Logging, QueueLogger, TranscriptionError
├── utils.py          # Timestamp formatting, file writing, media duration
├── cuda.py           # Windows CUDA/cuDNN path helpers
├── transcriber.py    # Core transcription logic (runs in background thread)
├── diarizer.py       # Speaker diarization via pyannote.audio
├── youtube.py        # YouTube audio download via yt-dlp
├── summarizer.py     # AI summarisation via OpenAI-compatible API
└── gui.py            # Tkinter UI
main.py               # Entry point
```

## Logs

Each session writes a rotating log file to `~/LocalTranscriptLogs/local_transcript_<timestamp>.log`. The path is shown in the app console on startup and in any error dialogs.
