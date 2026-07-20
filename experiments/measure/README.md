# MaCoDeR — Measurement suite (thesis Chapter 6)

Reproducible scripts that produce the **real** numbers for the Evaluation chapter.
All run CPU-only, from the project root, in `vision_env`:

```
./vision_env/Scripts/python.exe -m experiments.measure.measure_latency          # §6.7
./vision_env/Scripts/python.exe -m experiments.measure.eval_models              # §6.2
./vision_env/Scripts/python.exe -m experiments.measure.fusion_robustness        # §6.4 / §6.5
./vision_env/Scripts/python.exe -m experiments.measure.build_aligned_multimodal 10  # builds §6.3 data
./vision_env/Scripts/python.exe -m experiments.measure.ablation                 # §6.3
./vision_env/Scripts/python.exe -m experiments.measure.retrain_temporal         # §5.4.5 temporal LSTM
./vision_env/Scripts/python.exe -m scripts.train_audio_emotion_v2               # upgraded audio-emotion model
```

Each writes a JSON (and the eval writes `confusion_*.png`). Numbers below are from
the run on 2026-07-08 — re-run to regenerate.

---

## §6.7 Real-time latency — `measure_latency.py`

Drives the **actual deployed pipeline** (`process_frame` → `process_audio` →
`fuse_results`, the same functions the websocket route calls) with 120 real
RAVDESS video frames + audio. First (warm-up) call excluded.

| Stage | Mean (ms) | p95 (ms) |
|---|---|---|
| Visual pipeline | 178.1 | 274.7 |
| Audio pipeline | 63.1 | 348.5 |
| Fusion + response | 0.1 | 0.1 |
| **End-to-end per frame** | **241.3** | **470.9** |

N = 120 frames, CPU-only (no GPU), measured on the **final deployed build** (upgraded
audio-emotion model + timing modality + temporal LSTM wired in). **This substantiates the
"< 2 s" latency claim** (mean 0.24 s, p95 0.47 s end-to-end).

---

## §6.2 Per-model held-out evaluation — `eval_models.py`

Honest re-evaluation with **5-fold cross-validation** (mean ± std accuracy, macro-F1),
same estimator family as deployed (RandomForest + StandardScaler). Confusion matrices
saved as `confusion_*.png`.

| Model | Accuracy (mean ± std) | Macro-F1 | N | Classes | Protocol |
|---|---|---|---|---|---|
| Audio emotion (old exp_007, 5 collapsed feats) | 0.349 ± 0.027 | 0.34 | 1440 | 8 | 5-fold stratified |
| **Audio emotion — DEPLOYED (exp_009, 56 feats)** | **0.412** (SI) / 0.606 (rand) | 0.38 / 0.59 | 1440 | 8 | by-actor / stratified |
| Audio stress (random split) | 0.639 ± 0.026 | 0.63 | 1440 | 3 | 5-fold stratified |
| **Audio stress (subject-independent)** | **0.531 ± 0.021** | **0.52** | 1440 | 3 | 5-fold by actor |
| Visual behaviour (emotion) | 0.871 ± 0.012 | 0.87 | 2758 | 8 | 5-fold stratified |
| Cognitive state | 1.00 ± 0.00 | 1.00 | **24** | 3 | 5-fold stratified |

**Upgraded deployed audio-emotion model (`scripts/train_audio_emotion_v2.py` → `exp_009`):**
The live model was the weak exp_007 above (5 features collapsing all 13 MFCCs into one mean).
It is now retrained on an **aligned, richer 56-feature set** (per-coefficient MFCC mean+std,
MFCC deltas, chroma, spectral shape) extracted by the *same* `AudioFeatureExtractor` the live
pipeline uses (so no train/inference skew). Result: **random-split 0.61 / subject-independent
0.41** (macro-F1 0.59 / 0.38) — up from 0.34, same 8-class vocabulary. `audio_emotion_inference.py`
now loads `exp_009`. Report the **subject-independent 0.41** as the honest figure.

**Honest caveats to state in the thesis (each surfaced by this script):**

- **Subject-independent stress drops random-split 0.64 → 0.53.** The gap is speaker
  leakage: a random split lets the same actor appear in train and test. Report the
  **0.53** subject-independent number as the honest generalisation figure.
- **Audio emotion is genuinely hard (0.35, 8-class).** This matches the deployed
  `exp_007` model (~0.34) and is consistent with acted-speech literature — the
  contribution is the adaptive fusion + system, not state-of-the-art per-modality
  accuracy.
- **Visual behaviour 0.87 came *down* from 0.91 once the leaky `emotion_score`
  column (a 1:1 encoding of the label) was removed.** A residual optimism remains:
  the visual CSV has **no clip/actor id**, so a subject-independent split is not
  possible and frames from one clip may fall in both train and test — treat 0.87 as
  an **upper-bound / optimistic** number and note this limitation.
- **Cognitive state "1.00" is NOT meaningful — n = 24.** Report it as *illustrative
  only*; this is the honest replacement for the misleading headline flagged in the
  revision spec (§6.2). A larger labelled set is future work.

> These replace/confirm the pre-filled §6.2 values with reproducible ones. Cite the
> `confusion_*.png` figures.

---

## §6.4 / §6.5 Quality-adaptive vs fixed fusion — `fusion_robustness.py`

A **mechanism demonstration** (not an accuracy benchmark — RAVDESS has no aligned,
labelled *degraded* multimodal set, so a fixed-vs-adaptive accuracy table would be
fabricated). It drives the real `MultimodalCognitiveFusion` with a deliberate
disagreement (face='neutral' 0.85 vs audio='angry' 0.80) and degrades each channel:

| Condition | Vq | Aq | w_visual | w_audio | Adaptive | Fixed 50/50 | Warning |
|---|---|---|---|---|---|---|---|
| Clean | 0.90 | 0.70 | 0.577 | 0.423 | neutral | neutral | no |
| Degraded video (dark/blur) | 0.20 | 0.70 | 0.233 | 0.767 | **angry** | neutral | no |
| Degraded audio (noisy/quiet) | 0.90 | 0.15 | 0.864 | 0.136 | neutral | neutral | no |
| Video only (audio not routed) | 0.90 | 0.00 | 1.000 | 0.000 | neutral | neutral | no |
| Both channels poor | 0.20 | 0.15 | 0.586 | 0.414 | neutral | neutral | **YES** |

**Reading:** as the video quality falls (row 2) the adaptive weight shifts onto the
reliable channel (audio) and the fused label follows it, whereas the fixed 50/50
baseline never adapts; the "Poor signal quality" warning fires only when **both**
channels are poor (row 5) — i.e. single-modality operation is not falsely flagged.
This is the core novelty (Sub-RQ2 / Theme 3), demonstrated on the real deployed code.

---

## §6.3 Cross-modal ablation — `build_aligned_multimodal.py` + `ablation.py`

An **aligned per-utterance multimodal RAVDESS dataset** is now built (audio + video +
timing + label, joined by RAVDESS code), enabling a genuine ablation. Subject-independent
(by-actor) 5-fold CV, 1200 utterances, 10 actors, 8 emotions (chance = 0.125):

| Configuration | Accuracy (mean ± std) | Macro-F1 | Macro AUC | Features |
|---|---|---|---|---|
| Audio only | 0.313 ± 0.040 | 0.274 | 0.699 | 72 |
| Video only | 0.319 ± 0.057 | 0.295 | 0.735 | 10 |
| Timing only | 0.167 ± 0.011 | 0.158 | 0.570 | 5 |
| **Concatenation fusion (all)** | **0.430 ± 0.057** | **0.409** | **0.793** | 87 |

**This is the modality-contribution proof (§6.3):** fusion beats **every** unimodal
baseline (+0.11 accuracy over the best single modality; AUC 0.735 → 0.793). Video is the
most informative single modality, timing the weakest (but above chance), and combining all
three wins — exactly the contribution the proposal claims. The **timing modality is now
implemented** (`ai_engine/features/audio/timing_features.py`, VAD response-latency / pause /
speech-rate) and wired live.

## §5.4.5 Temporal-emotion LSTM — `retrain_temporal.py`

The original temporal LSTM was inflated by (a) the leaky `emotion_score` feature and
(b) a random split of overlapping sliding-window sequences. Retrained honestly:

| Config | Accuracy | Macro-F1 |
|---|---|---|
| Leaky replication (9 feat, random split) | 0.709 | 0.698 |
| **Honest (7 genuine feat, blocked split)** | **0.417** | **0.410** |

The 0.29 gap **is** the leakage. The honest 7-feature model (`temporal_emotion_model_v2.h5`)
is now **wired into the live pipeline** (`temporal_emotion_inference.py`): a rolling 60-frame
buffer feeds the LSTM so `temporal_emotion` is a real sequence prediction (verified: differs
from the frame emotion once the buffer fills), not the old passthrough.

## Still genuinely missing (future work — not fabricated here)

- **Learned `QualityAdaptiveFusion` net (Code Listing 3)** — the deployed mechanism is the
  deterministic quality-adaptive weighting (demonstrated in `fusion_robustness.py`); the
  *trained* net is future work.
- **Fixed-vs-adaptive *accuracy* under degradation (full §6.4 table)** — would need an
  aligned, labelled **degraded** multimodal set (systematically dimmed video + noisy audio
  per utterance). The mechanism is demonstrated; a labelled degraded-accuracy benchmark is
  future work.
