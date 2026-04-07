# Local Transcript Studio

A desktop application for transcribing local audio and video files entirely offline using [faster-whisper](https://github.com/SYSTRAN/faster-whisper). No data leaves your machine.

## Features

- Transcribes MP4, MKV, MOV, AVI, WebM, MP3, WAV, M4A, and FLAC files
- Outputs a plain-text transcript (`_transcript.txt`) and a timed subtitle file (`_subtitles.srt`) alongside the source file
- GPU acceleration via CUDA with automatic fallback to CPU
- Real-time progress bar showing percentage, elapsed time, ETA, and transcription speed
- Stop button to cancel an in-progress transcription
- Drag-and-drop file input (requires `tkinterdnd2`)
- Per-session log files written to `~/LocalTranscriptLogs/`

## Requirements

- Python 3.10+
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [tkinterdnd2](https://github.com/pmgagne/tkinterdnd2) _(optional — enables drag and drop)_
- [FFmpeg](https://ffmpeg.org/) with `ffprobe` on `PATH` _(optional — enables ETA calculation)_

For GPU acceleration, a CUDA-capable GPU with cuDNN 9 installed is required.

## Installation

```bash
pip install faster-whisper tkinterdnd2 yt-dlp
```

### FFmpeg (optional)

FFmpeg provides two things: `ffprobe` for ETA calculation during transcription, and the audio extraction pipeline used when downloading YouTube videos.

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
```

1. Drag a media file onto the drop zone, or click **Browse MP4 / Media**
2. Select a model (`medium` or `large-v3`) and preferred device (`cuda` or `cpu`)
3. Click **Start Transcription**
4. Output files are written to the same folder as the source file

To cancel a running transcription, click **Stop**. Any segments already processed are discarded and no output files are written.

## Models

| Model | Speed | Quality |
|-------|-------|---------|
| `medium` | Faster | Good |
| `large-v3` | Slower | Best |

Models are downloaded automatically by faster-whisper on first use and cached locally.

## Output files

Given an input file `interview.mp4`, two files are created in the same directory:

- `interview_transcript.txt` — plain text, one paragraph per segment
- `interview_subtitles.srt` — standard SRT format with timestamps

## Project structure

```
src/
├── constants.py      # Shared constants
├── logging_setup.py  # Logging, QueueLogger, TranscriptionError
├── utils.py          # Timestamp formatting, file writing, media duration
├── cuda.py           # Windows CUDA/cuDNN path helpers
├── transcriber.py    # Core transcription logic (runs in background thread)
└── gui.py            # Tkinter UI
main.py               # Entry point
```

## Logs

Each session writes a rotating log file to `~/LocalTranscriptLogs/local_transcript_<timestamp>.log`. The path is shown in the app console on startup and in any error dialogs.
