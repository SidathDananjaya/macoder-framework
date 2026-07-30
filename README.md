# MaCoDeR — Multimodal Cognitive-Stress & Deception-Risk Estimation

**MaCoDeR is a browser-based, quality-adaptive multimodal decision-support prototype
for estimating cognitive-stress indicators and possible deception-risk cues in remote
interactions.** It reads a standard webcam + microphone, fuses visual, audio and timing
signals in real time, and produces explainable indicators, an aggregated session report,
and a local-LLM interpretation.

> ⚠️ **It is not a lie detector.** It produces behavioural **indicators and risk scores
> only** — it does **not** determine truthfulness. Outputs must never be the sole basis
> for hiring, disciplinary, medical, legal or academic-integrity decisions, and require
> qualified human review. Deception risk is an **unvalidated indicator**.

Author: A.S.S.D Mendis (W2151433) · MSc Advanced Software Engineering, University of Westminster.

---

## What it does

- **Live capture** — the React frontend streams webcam frames + microphone audio to the
  FastAPI backend over a WebSocket.
- **Visual pipeline** — MediaPipe FaceMesh → EAR/blink, gaze, head pose, movement; DeepFace
  facial emotion; a sequence **LSTM** for temporal emotion.
- **Audio pipeline** — MFCC/prosodic features → audio emotion + audio stress; RMS speech
  intensity.
- **Timing modality** — energy-VAD response-latency / pause / speech-rate features.
- **Quality-adaptive fusion** — real-time video/audio **signal-quality scores** scale each
  modality's weight, so a degraded channel is down-weighted and a *"Poor signal quality"*
  warning appears when no channel is reliable.
- **Scoring** — recent-window cognitive-stress level, a 0–100 deception-**risk** score, and a
  cognitive state.
- **Reporting** — per-frame session recording (JSON/CSV), an aggregated report (stress/emotion/
  risk timelines, blink/gaze stats), and a local-LLM (Ollama) interpretation.

```
React (LiveCamera)  ──WebSocket──▶  FastAPI backend
                                      │
                      ┌───────────────┼────────────────┐
                   Visual AI        Audio AI         Timing AI
                      └───────────────┼────────────────┘
                            Quality-adaptive fusion
                                      │
                     Stress · Deception-risk · Cognitive state
                                      │
                 Dashboard · Session recording · Report · LLM
```

---

## Repository layout

| Path | Contents |
|---|---|
| `ai_engine/` | The AI engine: `features/` (visual, audio, quality, behavioral), `fusion/`, `temporal/`, `inference/`, `cognition/`, `analytics/` (logger, report, LLM), `models/`, `training/`, plus `configs/`, `data/`, `utils/` support modules |
| `backend/` | FastAPI app — routes, websocket stream, realtime engine/service layer |
| `frontend/` | React + Vite dashboard |
| `experiments/` | Trained models per experiment + **`measure/`** (Chapter-6 measurement suite) |
| `scripts/` | Dataset builders + model trainers |
| `datasets/` | `raw/RAVDESS` (training data) + `processed/` feature sets |
| `outputs/session_logs/` | Saved session recordings, CSVs and reports |

---

## Prerequisites

### Hardware

| Resource | Minimum / notes |
|---|---|
| **CPU** | Any modern x86-64 CPU. The full pipeline runs in real time on CPU — **no GPU required** (measured ~241 ms mean end-to-end latency, CPU-only). |
| **RAM** | 8 GB minimum, 16 GB recommended (TensorFlow + DeepFace + MediaPipe loaded together). |
| **Disk** | ~3 GB for the Python environment and bundled models. An additional ~13 GB only if you download the raw RAVDESS dataset to retrain. |
| **Webcam + microphone** | Required for a live session. A virtual camera/mic can be used to feed a recorded clip (see §5). |
| **OS** | Developed and tested on **Windows 11**. Linux/macOS should work with the same pinned stack (paths in commands below use Windows form). |

### Software

| Requirement | Version / notes |
|---|---|
| **Python** | **3.11** (developed on 3.11.9). TensorFlow 2.15 requires `numpy < 2`, so 3.11 is important — 3.12+ may not resolve the pinned stack. |
| **Node.js** | 18+ (for the Vite frontend). |
| **Ollama** | Optional — only for the AI Interpretation. Local install from [ollama.com](https://ollama.com); the app degrades gracefully without it. |
| **RAVDESS dataset** | Only needed to **train/retrain** models, not to run the live app (trained models ship under `experiments/`). Place under `datasets/raw/RAVDESS/` (`Audio_Speech_Actors_01-24/` + `Video_Speech_Actors_01-24/`). See [zenodo.org/record/1188976](https://zenodo.org/record/1188976). |

The core ML stack is: `tensorflow==2.15.0`, `keras==2.15.0`, `deepface==0.0.79`,
`mediapipe==0.10.9`, `opencv-python==4.8.1.78`, `numpy==1.26.4`, `librosa==0.11.0`,
`scikit-learn==1.8.0`, `fastapi`, `uvicorn`, `soundfile`.

> **Two environments.** `requirements.txt` (project root) is the **runtime** lock — the
> environment that serves the live system, and the one the Setup steps below build. The
> models were **trained** in a separate, newer environment locked in
> [`backend/requirements.txt`](backend/requirements.txt) (deepface 0.0.100, pandas 3.0.2,
> torch, shap). You only need that second one to retrain.

---

## Setup & run

### 1. Python environment

Create a fresh virtual environment and install the pinned dependencies. `requirements.txt`
is a lock of the known-good environment (numpy 1.26.4 / tensorflow 2.15.0 / deepface 0.0.79 …):
```bash
py -3.11 -m venv .venv
# Windows (PowerShell):  .\.venv\Scripts\Activate.ps1
# Windows (Git Bash):    source .venv/Scripts/activate
# macOS/Linux:           source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> ℹ️ The pins matter: TensorFlow 2.15 requires `numpy < 2`, so install into a **Python 3.11**
> environment. Newer numpy/tensorflow combinations will not resolve the pipeline.

### 2. Backend (FastAPI, `http://127.0.0.1:8000`)

With the environment activated, from the project root:
```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
The frontend is hard-wired to `http://127.0.0.1:8000` (REST) and `ws://127.0.0.1:8000/ws/live`
(WebSocket), so **keep this host/port**. CORS is open (`allow_origins=["*"]`) for local dev.

### 3. Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev            # serves the dashboard (Vite dev server)
```
Open the dashboard, grant **camera + microphone** permission, click **Start Session**, then
**Stop → Generate Report → AI Interpretation**.

### 4. AI Interpretation (Ollama — optional)

The Phase-9 interpreter calls a **local** Ollama server. Defaults (override with env vars):

```bash
# defaults: OLLAMA_HOST=http://localhost:11434  OLLAMA_MODEL=qwen3.5:9b
ollama serve
ollama pull qwen3.5:9b          # or set OLLAMA_MODEL to a model you have, e.g. llama3.1:8b
```
If Ollama is not running or the model is missing, the interpreter returns a graceful fallback
instead of crashing — the rest of the system is unaffected.

### 5. Testing with YouTube videos (no live subject)

Route a clip's picture + sound into a virtual camera/mic (e.g. OBS Virtual Camera) so the
deployed system treats it as a live webcam session. Select the virtual devices when the browser
prompts for camera/microphone, then run a session as normal. The five clips used for system
testing and their results are summarised in Appendix C of the report and under `test-results/`.

---

## Training / retraining models (optional)

All trained models are already included under `experiments/`, so training is **not required
to run the system**. To reproduce or retrain (requires the RAVDESS dataset, with the venv
activated):

```bash
# aligned audio-emotion model (deployed):
python -m scripts.train_audio_emotion_v2
# honest temporal-emotion LSTM (deployed):
python -m experiments.measure.retrain_temporal
# Chapter-6 measurement suite (latency / eval / ablation / fusion):
#   see experiments/measure/README.md for the full command list
```

---

## Documentation

| File | Purpose |
|---|---|
| [`experiments/measure/README.md`](experiments/measure/README.md) | Chapter-6 measurement suite + reproducible results |
| `test-results/` | System-testing screenshots for the five evaluation clips (Appendix C) |

---

## Test accounts / credentials

**None required.** The system has no login or authentication layer — open the dashboard and
start a session directly. There are no user accounts, passwords, or API keys to configure.

---

## Known limitations

These are the technical limitations that qualify the results (full treatment in Chapter 7,
§7.4 of the report):

- **Acted, not spontaneous, training data.** Models are trained on RAVDESS, where actors
  portray emotion on cue; portrayed affect is not experienced affect.
- **No deception/stress ground truth in real-world testing.** The YouTube interview clips have
  no verifiable labels, so that testing assesses behaviour, robustness and explainability —
  **not** deception or stress accuracy.
- **Small cognitive-state evaluation set** (n = 24), so the cognitive model's headline figure
  is not a robust generalisation estimate.
- **Demographic bias not audited.** RAVDESS is a narrow actor pool; formal fairness auditing
  across skin tone, gender, age and accent is future work.
- **Deception risk is an unvalidated indicator** — not a lie detector (see below).

---

## Ethical statement

This system produces behavioural **indicators and risk scores only**. It does **not**
determine truthfulness, and must not be used as the sole or primary basis for hiring,
disciplinary, medical, legal or academic-integrity decisions. All outputs require qualified
human review and formal validation before any real-world use.
