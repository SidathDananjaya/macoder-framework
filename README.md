# MaCoDeR: Multimodal Cognitive-Stress and Deception-Risk Estimation

MaCoDeR estimates cognitive-stress indicators and possible deception-risk cues from a live
webcam and microphone, in the browser, and explains what it saw rather than just returning a
number.

Most stress or deception tools give a single score with nothing behind it, which leaves the
person reading it no way to judge whether to believe it. MaCoDeR pairs every session with
evidence: per-frame visual, audio and timing measurements, a signal-quality score for each
channel, a timeline of how stress and emotion moved during the session, and a written
interpretation produced by a local LLM. When the signal is too poor to say anything useful, it
says so instead of guessing. It combines MediaPipe, DeepFace and a temporal LSTM on the visual
side with MFCC and prosodic models on the audio side, served through a FastAPI backend and a
React web interface.

This repository is the implementation accompanying the MSc thesis *"Multimodal Cognitive-Stress
and Deception-Risk Estimation for Remote Interactions"*.

> **This is not a lie detector.** It produces behavioural indicators and risk scores, and it
> does not determine whether anyone is telling the truth. Its outputs must never be the sole
> basis for hiring, disciplinary, medical, legal or academic-integrity decisions, and they
> always need review by a qualified person. Treat the deception-risk score as an unvalidated
> indicator.

Author: A.S.S.D Mendis (W2151433), MSc Advanced Software Engineering, University of Westminster.

## Contents

1. [Requirements](#requirements)
2. [Languages, libraries and frameworks](#languages-libraries-and-frameworks)
3. [Installation](#installation)
4. [Model artifacts](#model-artifacts)
5. [Dataset preparation](#dataset-preparation)
6. [Configuration](#configuration)
7. [Running the system](#running-the-system)
8. [API reference](#api-reference)
9. [Session storage](#session-storage)
10. [Training and evaluation](#training-and-evaluation)
11. [Results](#results)
12. [Testing](#testing)
13. [Default credentials](#default-credentials)
14. [External services and API keys](#external-services-and-api-keys)
15. [Troubleshooting](#troubleshooting)
16. [Known limitations](#known-limitations)
17. [Ethical statement](#ethical-statement)
18. [Repository contents](#repository-contents)
19. [Citation](#citation)

There is no Docker image and no container orchestration. The system runs directly on Python and
Node.

## Requirements

### Software

* Python 3.11. Development used 3.11.9.
* Node.js 18 or newer, with npm.
* Ollama, optional, only for the AI Interpretation feature.

Development and testing were done on Windows 11. Linux and macOS should work with the same
pinned versions, though the commands below are written in Windows form.

Python 3.11 is not a loose suggestion. TensorFlow 2.15 requires `numpy < 2`, and Python 3.12 and
later will not resolve the pinned stack.

### Hardware

* CPU only. No GPU is required, either to run the system or to retrain it. Mean end-to-end
  latency measured 241 ms per frame on CPU.
* 8 GB RAM minimum, 16 GB recommended. TensorFlow, DeepFace and MediaPipe are all held in memory
  at once.
* About 3 GB of disk for the Python environment. The repository itself, including every trained
  model, is around 130 MB.
* A webcam and a microphone, for a live session. A virtual camera and mic can substitute, which
  is how the recorded-clip testing was done.
* Add roughly 13 GB of disk only if you download the raw RAVDESS dataset to retrain.

## Languages, libraries and frameworks

| Layer | Technology | Version |
|---|---|---|
| Backend language | Python | 3.11.9 |
| Frontend language | JavaScript (ES2022) | |
| Deep learning | TensorFlow | 2.15.0 |
| Model API | Keras | 2.15.0 |
| Face mesh and landmarks | MediaPipe | 0.10.9 |
| Facial emotion | DeepFace | 0.0.79 |
| Computer vision | OpenCV (`opencv-python`) | 4.8.1.78 |
| Audio analysis | librosa | 0.11.0 |
| Audio I/O | soundfile, sounddevice | 0.13.1, 0.5.5 |
| Classical models | scikit-learn | 1.8.0 |
| Numerics | NumPy, pandas | 1.26.4, 2.1.4 |
| Web framework | FastAPI | 0.136.1 |
| ASGI server | Uvicorn | 0.46.0 |
| Validation | Pydantic | 2.13.4 |
| Frontend framework | React | 19.2 |
| Frontend build | Vite | 8.0 |
| Charts | Recharts | 3.8 |
| Styling | Tailwind CSS | 4.2 |
| HTTP client | axios | 1.16 |
| Local LLM | Ollama, default model `qwen3.5:9b` | |
| Research dashboard | Streamlit | 1.58.0 |

Runtime versions are pinned in `requirements.txt`. Frontend versions resolve from
`frontend/package-lock.json`.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/SidathDananjaya/macoder-framework.git
cd macoder-framework
```

### 2. Create the Python environment

```bash
py -3.11 -m venv .venv

# Windows (PowerShell):  .\.venv\Scripts\Activate.ps1
# Windows (Git Bash):    source .venv/Scripts/activate
# macOS/Linux:           source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` is a lock of the known-good runtime environment, built around numpy 1.26.4,
tensorflow 2.15.0 and deepface 0.0.79. Install it into Python 3.11, for the reason given under
[Requirements](#requirements).

There are two environments in this project and they are not interchangeable:

| Environment | Locked in | Purpose |
|---|---|---|
| Runtime | `requirements.txt` | Serves the live system. This is what you want. |
| Training | `backend/requirements.txt` | The environment the models were trained in. Newer deepface (0.0.100), pandas 3.0.2, plus torch and shap. |

You only need the second one to retrain. See [Training and evaluation](#training-and-evaluation).

### 3. Install the frontend

```bash
cd frontend
npm install
cd ..
```

### 4. Install Ollama, optional

Only needed for the AI Interpretation step. Install from [ollama.com](https://ollama.com), then:

```bash
ollama serve
ollama pull qwen3.5:9b
```

Without it, everything else still works. The interpreter falls back to a rule-based summary.

### 5. Check the models are present

```bash
ls experiments/exp_009_audio_emotion_v2/audio_emotion_model.pkl
ls experiments/exp_006_temporal_emotion/temporal_emotion_model_v2.h5
```

Both should exist in a fresh clone. Nothing else needs downloading. See
[Model artifacts](#model-artifacts).

## Model artifacts

Every trained model is committed to this repository, about 98 MB in total. There is no external
download step and no Google Drive bundle to fetch. A fresh clone can run a full session
immediately.

| Experiment | Artifact | Size | Used at run time |
|---|---|---:|---|
| `exp_001_audio_baseline` | `audio_baseline_model.pkl` | 18 MB | Yes |
| `exp_002_stress_classifier` | `stress_classifier.pkl` | 11 MB | Yes |
| `exp_002_visual_baseline` | visual baseline model | | Yes |
| `exp_005_visual_behavior` | `visual_behavior_model.pkl` | 13 MB | Yes |
| `exp_006_temporal_emotion` | `temporal_emotion_model_v2.h5` | 428 KB | Yes |
| `exp_007_audio_emotion` | `audio_emotion_model.pkl` | 27 MB | Superseded by exp_009 |
| `exp_008_cognitive` | `cognitive_model.pkl` | 252 KB | Yes |
| `exp_009_audio_emotion_v2` | `audio_emotion_model.pkl` | 28 MB | Yes, the deployed audio-emotion model |
| `exp_003`, `exp_004` | early fusion and quality-adaptive experiments | | No, kept for the record |

Each folder also carries its `scaler.pkl`, `label_encoder.pkl` and a `metrics.json` recording
what that run scored.

DeepFace is the one exception. It downloads its own facial-expression weights, 5.8 MB, from
GitHub on first use and caches them under `~/.deepface/weights/`. That needs a network connection
the first time you run a session, and never again.

## Dataset preparation

The dataset is only needed to retrain. Skip this section to run the system.

MaCoDeR uses RAVDESS, the Ryerson Audio-Visual Database of Emotional Speech and Song: 24 actors
performing eight emotion categories, with matched audio and video recordings.

The raw dataset is not in this repository. It is about 13 GB and `datasets/raw/` is deliberately
ignored by git. Download it from
[https://zenodo.org/records/1188976](https://zenodo.org/records/1188976).

### Required layout

```
datasets/raw/RAVDESS/
├── Audio_Speech_Actors_01-24/
└── Video_Speech_Actors_01-24/
```

`ai_engine/configs/project_paths.py` resolves these paths, so the folder names must match
exactly.

### Already prepared

The engineered feature sets are committed, about 17 MB, under `datasets/processed/`. They are
what the training scripts actually read, so several retraining steps work without the raw
dataset in place.

## Configuration

There is no `.env` file and no configuration file to edit. The system reads three environment
variables, all of them optional, and all of them for the LLM interpreter. They are defined in
`ai_engine/analytics/llm_interpreter.py`.

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Where the local Ollama server is listening |
| `OLLAMA_MODEL` | `qwen3.5:9b` | Which pulled model to use for the interpretation |
| `OLLAMA_THINK` | `false` | Set `true` to let the model emit its reasoning before answering |

Everything else is fixed in code. Filesystem paths resolve against the repository root through
`ai_engine/configs/project_paths.py`, and the host and port are passed on the uvicorn command
line.

To use a model you already have rather than pulling `qwen3.5:9b`:

```bash
# PowerShell
$env:OLLAMA_MODEL = "llama3.1:8b"

# Git Bash
export OLLAMA_MODEL=llama3.1:8b
```

## Running the system

Open two terminals. The backend must be running before the frontend is useful.

```bash
# Terminal 1: the API on port 8000, from the repository root, venv activated
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

```bash
# Terminal 2: the web interface on port 5173
cd frontend
npm run dev
```

Keep that host and port. The frontend has `http://127.0.0.1:8000` and
`ws://127.0.0.1:8000/ws/live` written into `frontend/src/services/api.js` and
`frontend/src/components/LiveCamera.jsx`, so changing the backend port means editing both.

| What | Address |
|---|---|
| Web interface | http://localhost:5173 |
| API | http://127.0.0.1:8000 |
| Interactive API documentation | http://127.0.0.1:8000/docs |

Check the API is up:

```bash
curl http://127.0.0.1:8000/
```

It replies `{"message": "MaCoDeR API Running"}`.

### Using the web interface

Open http://localhost:5173 and allow camera and microphone access when the browser asks. Then:

1. Press **Start Session**. Live charts begin filling as frames are processed.
2. Talk, or let the subject talk, for at least 30 seconds. Short sessions produce thin reports.
3. Press **Stop**. The recording is written to `outputs/session_logs/` automatically.
4. Press **Generate Report** for the aggregated timelines and statistics.
5. Press **AI Interpretation** for the written summary, if Ollama is running.

The first frame is slow, because TensorFlow, MediaPipe and DeepFace all initialise on the first
request. After that it settles to roughly 241 ms per frame.

If a channel degrades, for example the room goes dark or the mic is muted, that modality is
down-weighted automatically. If neither channel is usable you get a "Poor signal quality"
warning rather than a fabricated score.

### Testing with a recorded video instead of a live subject

Route a clip's picture and sound into a virtual camera and microphone, using something like OBS
Virtual Camera, so the system sees an ordinary webcam session. Pick the virtual devices when the
browser asks for permission, then run a session as normal. The five clips used for system
testing are summarised in Appendix C of the report, with screenshots under `test-results/`.

### Research dashboard, optional

A small Streamlit dashboard reads the saved session CSVs for offline inspection. It is separate
from the React interface and is not part of the main workflow.

```bash
streamlit run dashboard/app.py
```

## API reference

Base address `http://127.0.0.1:8000`. Interactive documentation at `/docs`. There is no
authentication on any endpoint.

All REST endpoints are mounted under `/api`. The WebSocket and the root health message are not.

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/` | | Liveness message |
| WS | `/ws/live` | JSON frames | Per-frame analysis, streamed |
| POST | `/api/predict-stress` | multipart, field `file` | Stress prediction for one audio file |
| GET | `/api/session-data` | | Every row of the latest session CSV |
| GET | `/api/session-summary` | | Aggregate statistics for the latest session CSV |
| GET | `/api/session/status` | | Progress of the in-flight recording |
| POST | `/api/session/export` | | Write the flat CSV log |
| POST | `/api/session/export/json` | | Write the structured JSON recording |
| GET | `/api/session/recordings` | | List saved recording filenames |
| GET | `/api/session/recordings/latest` | | The most recent recording, in full |
| GET | `/api/session/report` | | Aggregated report for the current session |
| POST | `/api/session/report/export` | | Write that report to disk |
| GET | `/api/session/report/{filename}` | | Rebuild a report from a stored recording |
| POST | `/api/session/interpret` | | LLM interpretation of the current session |
| POST | `/api/session/interpret/{filename}` | | LLM interpretation of a stored recording |

### WS /ws/live

The main channel. Opening the socket resets the engine, the audio buffer, the session log and
the session memory, so one connection is one session.

Send one JSON object per frame:

```json
{
  "frame": "data:image/jpeg;base64,/9j/4AAQSkZJRgABA...",
  "audio": [0.0021, -0.0044, 0.0139],
  "sample_rate": 48000
}
```

`frame` is a JPEG data URL at quality 0.7, produced by the canvas in `LiveCamera.jsx`. `audio` is
an array of raw float samples collected since the previous frame. `sample_rate` comes from the
browser's `AudioContext` and defaults to 22050 if absent.

The reply merges the visual, audio and fusion results:

```json
{
  "emotion": "neutral",
  "visual_emotion_confidence": 0.62,
  "temporal_emotion": "calm",
  "final_emotion": "neutral",
  "stress_level": "LOW",
  "visual_stress": 0.18,
  "cognitive_load": "LOW",
  "deception_risk": 12.0,
  "visual_cognitive_state": "ATTENTIVE",
  "cognitive_confidence": 0.71,
  "fusion_confidence": 0.66,
  "temporal_state": "stable",
  "blink_count": 7,
  "movement_score": 0.04,
  "gaze_direction": "center",
  "gaze_stability": 0.88,
  "gaze_shift_frequency": 0.12,
  "video_quality": 0.81,
  "head_pose": { "yaw": -2.1, "pitch": 4.8, "roll": 0.3 },
  "audio_emotion": "neutral",
  "audio_confidence": 0.44,
  "audio_stress": "LOW",
  "audio_stress_confidence": 0.51,
  "speech_intensity": 0.07,
  "audio_quality": 0.73,
  "response_latency": 0.9,
  "pause_ratio": 0.22,
  "speech_rate": 3.4,
  "voiced_ratio": 0.61,
  "mean_pause": 0.35,
  "session_memory": { }
}
```

If a frame cannot be decoded, or no face is found, the engine returns a neutral fallback
(`"emotion": "unknown"`, `"stress_level": "LOW"`, `"video_quality": 0.0`) rather than dropping
the connection.

Closing the socket writes the session recording to `outputs/session_logs/` automatically.

### POST /api/predict-stress

Offline stress prediction for a single audio file, independent of the live session.

```bash
curl -X POST http://127.0.0.1:8000/api/predict-stress \
     -F "file=@sample.wav"
```

```json
{
  "filename": "sample.wav",
  "prediction": { "stress_level": "MEDIUM", "confidence": 0.58 }
}
```

The uploaded file is written to `temp_uploads/` before it is analysed.

### GET /api/session-summary

Aggregates the most recent session CSV.

```json
{
  "records": 412,
  "avg_emotion_confidence": 0.63,
  "avg_temporal_confidence": 0.59,
  "max_blink_rate": 34.0,
  "dominant_emotion": "neutral",
  "dominant_temporal_emotion": "calm",
  "most_common_stress": "LOW",
  "most_common_cognitive_load": "LOW",
  "most_common_deception": "LOW"
}
```

Returns `{"error": "No session CSV files found"}` when nothing has been recorded yet.

### GET /api/session/report

The aggregated report, built by `ai_engine/analytics/session_report_generator.py`.

| Field | Contents |
|---|---|
| `session_id`, `started_at`, `ended_at` | Session identity and wall-clock bounds |
| `frame_count`, `duration_seconds`, `duration` | Size of the recording |
| `stress` | Average percent, average and highest level, when the peak occurred, full distribution |
| `emotion` | Dominant emotion, distribution, stability percent, change timeline |
| `risk` | Average and highest deception-risk score, when the peak occurred, dominant level |
| `blink` | Blink counts and rate statistics |
| `gaze` | Gaze direction distribution and stability |

### POST /api/session/interpret

Generates the report, then sends it to Ollama for a written interpretation. This is the slowest
endpoint, since it waits on local LLM generation.

When Ollama is unreachable or the model is not pulled, it returns a rule-based summary along
with a message naming the missing model, rather than failing.

The `{filename}` variants of `report` and `interpret` work on a stored recording instead of the
live session. Both reject any filename containing a path separator with a 400, and return 404
when the recording does not exist.

## Session storage

MaCoDeR uses no database. There is no SQL, no ORM, no migrations and no database server.
Everything is flat files on disk.

| Property | Value |
|---|---|
| Location | `outputs/session_logs/` |
| Formats | CSV, one row per frame, and JSON, the structured recording |
| Written by | `backend/app/services/session_logger_service.py` |
| Naming | `{session_id}.csv`, `{session_id}.json`, `{session_id}_report.json` |
| Lifecycle | Written when the WebSocket closes, or on demand through the export endpoints |
| In version control | No. `outputs/` is git-ignored and regenerates at run time |

The CSV is the flat per-frame log, convenient for pandas and for the Streamlit dashboard. The
JSON recording carries the same frames plus session metadata, and is what the report generator
and the LLM interpreter consume. Reports are derived, never authoritative, so deleting them
costs nothing as long as the recording survives.

`temp_uploads/` holds files posted to `/api/predict-stress` and is also git-ignored.

## Training and evaluation

Nothing here is needed to run the system. Every deployed model is already committed.

Retraining uses the second environment, not the runtime one:

```bash
py -3.11 -m venv backend/venv
backend/venv/Scripts/python.exe -m pip install -r backend/requirements.txt
```

That lock includes `--extra-index-url` for the CPU builds of torch, which newer DeepFace pulls
in. Installing it without that line fails.

### Retraining the deployed models

```bash
# aligned audio-emotion model, exp_009, the deployed one
python -m scripts.train_audio_emotion_v2

# temporal-emotion LSTM with the leakage-free split
python -m experiments.measure.retrain_temporal
```

Both need the RAVDESS dataset in place. See [Dataset preparation](#dataset-preparation).

### The measurement suite

`experiments/measure/` holds the scripts that produced the Chapter 6 numbers, along with their
JSON output and confusion matrices.

```bash
python -m experiments.measure.measure_latency          # end-to-end timing
python -m experiments.measure.eval_models              # per-model accuracy
python -m experiments.measure.ablation                 # modality ablation
python -m experiments.measure.fusion_robustness        # quality-adaptive behaviour
```

Each writes a `*_results.json` next to itself. `experiments/measure/README.md` has the full
command list and the discussion.

## Results

Measured on the final deployed build, CPU only. Full detail, including confusion matrices and
the caveats behind each figure, is in
[`experiments/measure/README.md`](experiments/measure/README.md).

### Latency

| Stage | Mean | 95th percentile |
|---|---:|---:|
| Visual pipeline | 178.1 ms | 274.7 ms |
| Audio pipeline | 63.1 ms | 348.5 ms |
| Fusion and response | 0.1 ms | 0.1 ms |
| **End to end, per frame** | **241.3 ms** | **470.9 ms** |

Measured over 120 frames. This is what supports the claim of real-time operation under two
seconds.

### Modality ablation

Subject-independent, 5-fold, 1200 utterances from 10 actors across 8 emotion classes.

| Configuration | Accuracy | Macro F1 | Macro AUC |
|---|---:|---:|---:|
| Audio only | 0.313 | 0.274 | 0.699 |
| Video only | 0.319 | 0.295 | 0.735 |
| Timing only | 0.167 | 0.158 | 0.570 |
| **All three, fused** | **0.430** | **0.409** | **0.793** |

Fusion beats every individual modality, which is the central claim the architecture rests on.

### Individual models

| Model | Accuracy | Split |
|---|---:|---|
| Audio emotion, deployed (exp_009) | 0.412 | Subject-independent, by actor |
| Audio emotion, deployed (exp_009) | 0.606 | Random stratified |
| Audio stress | 0.531 | Subject-independent |
| Audio stress | 0.639 | Random stratified |
| Visual behaviour | 0.871 | n = 2758 |
| Cognitive state | 1.000 | n = 24, see below |

Report the subject-independent figures. The random-split numbers are higher because the same
actor appears in both train and test, which flatters the model.

The cognitive-state figure of 1.000 is not a real accuracy claim. The evaluation set is 24
samples, which is far too small to generalise from. It is reported for completeness, not as
evidence. See [Known limitations](#known-limitations).

## Testing

There is no automated test suite, no pytest configuration and no continuous integration.

What exists is seven manual smoke scripts under `scripts/`, each exercising one part of the
pipeline directly. Run them from the repository root with the venv activated:

```bash
python -m scripts.test_visual_pipeline           # MediaPipe, landmarks, head pose
python -m scripts.test_blink_detection           # EAR and blink counting
python -m scripts.test_audio_pipeline            # end-to-end audio feature extraction
python -m scripts.test_audio_emotion_model       # audio-emotion model against saved features
python -m scripts.test_audio_emotion_realtime    # audio-emotion on live microphone input
python -m scripts.test_advanced_audio_features   # prosodic feature extraction
python -m scripts.test_phase3_audio_features     # timing and VAD features
```

They print results for a human to read rather than asserting, so they report nothing on success
and are not suitable for automation as they stand. Several need a working webcam or microphone.

System-level verification was done by running full sessions and recording the output. The
evidence is in `test-results/`: five YouTube interview clips and two live self-capture sessions,
two screenshots each.

## Default credentials

None. MaCoDeR has no authentication, no user accounts, no login page and no API keys. Open the
dashboard and start a session.

The system should not be exposed on a public network as it stands. CORS accepts any origin
(`allow_origins=["*"]` in `backend/app/main.py`), there is no rate limiting, and every endpoint
that reads or writes session recordings is open. It is built to run on one machine, bound to
`127.0.0.1`.

## External services and API keys

No API keys are required and there is nothing to sign up for.

| Service | When | Required |
|---|---|---|
| Ollama | Only for the AI Interpretation step. Runs locally on your own machine | No. Without it the interpreter falls back to a rule-based summary |
| GitHub, via DeepFace | Facial-expression weights, 5.8 MB, downloaded on first use and cached in `~/.deepface/weights/` | Network needed once, on the first session only |
| Zenodo | Downloading RAVDESS, only if you retrain | A one-off manual download. The link is public |

Nothing leaves the machine during a session. There is no telemetry, no cloud inference and no
third-party analytics. The LLM interpretation runs against a local Ollama server, so session
content is never sent to a hosted model.

## Troubleshooting

**The camera preview stays black, or the browser never prompts for permission**

Browsers only allow camera and microphone access over `localhost` or HTTPS. Open
http://localhost:5173, not a LAN IP address. Check that no other application is holding the
camera.

**The dashboard loads but no data appears after Start Session**

The backend is not reachable. Confirm terminal 1 is still running and answering:

```bash
curl http://127.0.0.1:8000/
```

The frontend hard-codes port 8000, so a backend started on a different port will never connect.

**The first frame takes several seconds**

Expected. TensorFlow, MediaPipe and DeepFace all initialise lazily on the first frame. Only the
first one is slow.

**AI Interpretation returns a rule-based summary instead of a written one**

Ollama is not running, or the model is not pulled. Check both:

```bash
ollama list
curl http://localhost:11434/api/tags
```

If the model you have is not `qwen3.5:9b`, set `OLLAMA_MODEL` to one you do have. See
[Configuration](#configuration).

**A "Poor signal quality" warning during a session**

Working as intended, not a fault. Both the video and audio quality scores fell below threshold,
so the system declines to report a score it cannot support. Improve the lighting, move closer to
the microphone, or reduce background noise.

**Report or interpretation says there are no recordings**

Nothing has been saved yet. Recordings are written when the WebSocket closes, so press Stop
before generating a report. Check what exists:

```bash
curl http://127.0.0.1:8000/api/session/recordings
```

**`No module named 'ai_engine'` or `No module named 'backend'`**

Run from the repository root, not from inside `backend/`. The uvicorn target
`backend.app.main:app` is an import path and resolves relative to the working directory.

**Installation fails resolving numpy or tensorflow**

You are on the wrong Python. TensorFlow 2.15 requires `numpy < 2`, so the environment must be
Python 3.11. Confirm with `python --version` inside the activated venv.

**Port 8000 or 5173 is already in use**

Vite will offer the next free port for the frontend. For the backend, changing the port also
means editing `frontend/src/services/api.js` and `frontend/src/components/LiveCamera.jsx`, which
have the address written into them.

## Known limitations

These qualify every result above. Chapter 7, section 7.4 of the report covers them in full.

**The training data is acted, not spontaneous.** The models are trained on RAVDESS, where actors
portray emotion on cue. Portrayed affect is not the same thing as experienced affect.

**Real-world testing has no ground truth for deception or stress.** The YouTube interview clips
carry no verifiable labels, so that testing tells you about behaviour, robustness and
explainability. It says nothing about deception or stress accuracy.

**The cognitive-state evaluation set is small** at n = 24, so that model's headline figure is not
a reliable estimate of how well it generalises.

**Demographic bias has not been audited.** RAVDESS draws on a narrow pool of actors, and formal
fairness auditing across skin tone, gender, age and accent remains future work.

**Deception risk is an unvalidated indicator**, not a lie detector.

## Ethical statement

This system produces behavioural indicators and risk scores. It does not determine whether
someone is telling the truth, and it must not be used as the sole or primary basis for hiring,
disciplinary, medical, legal or academic-integrity decisions. Every output needs review by a
qualified person, and the system would need formal validation before any real-world use.

## Repository contents

| Path | Contents |
|---|---|
| `ai_engine/features/` | Visual, audio, quality and behavioural feature extraction |
| `ai_engine/fusion/` | Confidence fusion, temporal decisions, cognitive-state generation |
| `ai_engine/temporal/` | Session memory and temporal smoothing |
| `ai_engine/inference/` | Real-time inference wrappers |
| `ai_engine/analytics/` | Session logger, report generator, LLM interpreter |
| `ai_engine/models/` | Model loading and SHAP explainability |
| `ai_engine/training/` | Trainers and evaluators |
| `ai_engine/configs/` | Project paths |
| `backend/app/api/routes/` | FastAPI route handlers |
| `backend/app/services/` | Real-time engines, fusion, session logging |
| `backend/app/websocket/` | Connection manager for the live stream |
| `frontend/src/components/` | LiveCamera, LiveCharts, SessionReport, SessionTimeline, SessionInterpretation |
| `frontend/src/services/` | API client |
| `experiments/exp_001` to `exp_009` | Trained models, scalers, encoders and metrics per experiment |
| `experiments/measure/` | Chapter 6 measurement scripts, results and confusion matrices |
| `scripts/` | Dataset builders, model trainers, manual smoke scripts |
| `datasets/processed/` | Engineered feature sets, committed |
| `datasets/raw/` | RAVDESS goes here, git-ignored |
| `dashboard/app.py` | Optional Streamlit dashboard for offline inspection |
| `outputs/session_logs/` | Session recordings, CSVs and reports, git-ignored |
| `test-results/` | System-testing screenshots, Appendix C |
| `requirements.txt` | Runtime environment lock |
| `backend/requirements.txt` | Training environment lock |

## Citation

Dataset:

> Livingstone, S. R., and Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional
> Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North
> American English. *PLoS ONE*, 13(5), e0196391.
> [https://zenodo.org/records/1188976](https://zenodo.org/records/1188976)

Built with [MediaPipe](https://github.com/google-ai-edge/mediapipe),
[DeepFace](https://github.com/serengil/deepface),
[TensorFlow](https://www.tensorflow.org/), [librosa](https://librosa.org/),
[FastAPI](https://fastapi.tiangolo.com/), [React](https://react.dev/) and
[Ollama](https://ollama.com).
