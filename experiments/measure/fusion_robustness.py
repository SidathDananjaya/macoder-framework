"""Quality-adaptive vs fixed fusion - robustness demonstration (Sections 6.4/6.5).

This is a **mechanism demonstration**, not an accuracy benchmark: RAVDESS has no
aligned, labelled *degraded* multimodal test set, so we cannot honestly report a
fixed-vs-adaptive accuracy table. Instead we drive the *real* deployed fusion
(`MultimodalCognitiveFusion`) with a controlled disagreement case and degrade
each channel's quality, showing that:

* the **adaptive** fusion shifts its modality weights onto the reliable channel
  as the other degrades, and follows it in the fused decision;
* a **fixed** 50/50 fusion (quality ignored) does not adapt;
* the **"Poor signal quality" warning** fires only when the best channel is poor.

Scenario: the face reads 'neutral' (conf 0.85) while the audio reads 'angry'
(conf 0.80) - a deliberate disagreement so the *winner* reveals which channel is
driving the decision.

Run:
    ./vision_env/Scripts/python.exe -m experiments.measure.fusion_robustness

Writes experiments/measure/fusion_robustness_results.json.
"""

import json
import os

from ai_engine.fusion.multimodal_cognitive_fusion import (
    MultimodalCognitiveFusion,
    QUALITY_WARNING_THRESHOLD,
)

OUT_PATH = "experiments/measure/fusion_robustness_results.json"

VISUAL_EMOTION, VISUAL_CONF = "neutral", 0.85
AUDIO_EMOTION, AUDIO_CONF = "angry", 0.80

CONDITIONS = [
    ("Clean (good video + good audio)", 0.90, 0.70),
    ("Degraded video (dark/blurred)", 0.20, 0.70),
    ("Degraded audio (noisy/quiet)", 0.90, 0.15),
    ("Video only (audio not routed)", 0.90, 0.00),
    ("Both channels poor", 0.20, 0.15),
]


def _fixed_5050(v_emo, v_conf, a_emo, a_conf):
    """Baseline: quality-agnostic 50/50 - just the more confident raw label."""
    return v_emo if v_conf >= a_conf else a_emo


def run():
    fusion = MultimodalCognitiveFusion()
    rows = []

    for name, vq, aq in CONDITIONS:
        # Adaptive: the real deployed fusion (quality scales each confidence).
        out = fusion.process(
            visual_emotion=VISUAL_EMOTION,
            visual_emotion_confidence=VISUAL_CONF,
            audio_emotion=AUDIO_EMOTION,
            audio_emotion_confidence=AUDIO_CONF,
            visual_stress="LOW", visual_stress_confidence=1.0,
            audio_stress="MEDIUM", audio_stress_confidence=0.8,
            blink_rate=3, head_movement=5,
            gaze_stability=1.0, gaze_shift_frequency=0,
            video_quality=vq, audio_quality=aq,
        )
        adaptive_emotion = out["fused_emotion"] if "fused_emotion" in out else out.get("final_emotion")
        weights = out["modality_weights"]

        rows.append({
            "condition": name,
            "video_quality": vq,
            "audio_quality": aq,
            "adaptive_weight_visual": weights["visual"],
            "adaptive_weight_audio": weights["audio"],
            "adaptive_fused_emotion": adaptive_emotion,
            "fixed_5050_emotion": _fixed_5050(
                VISUAL_EMOTION, VISUAL_CONF, AUDIO_EMOTION, AUDIO_CONF
            ),
            "warning_banner": out["quality_warning"],
        })

    result = {
        "scenario": {
            "visual_emotion": VISUAL_EMOTION, "visual_conf": VISUAL_CONF,
            "audio_emotion": AUDIO_EMOTION, "audio_conf": AUDIO_CONF,
            "warning_threshold": QUALITY_WARNING_THRESHOLD,
        },
        "conditions": rows,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print("=== Quality-adaptive vs fixed fusion (Sections 6.4/6.5) ===")
    print(f"Face='{VISUAL_EMOTION}'({VISUAL_CONF})  Audio='{AUDIO_EMOTION}'({AUDIO_CONF})  "
          f"warn<{QUALITY_WARNING_THRESHOLD}\n")
    hdr = f"{'Condition':<34}{'Vq':>5}{'Aq':>5}{'w_vis':>7}{'w_aud':>7}{'ADAPTIVE':>10}{'FIXED':>9}{'warn':>6}"
    print(hdr)
    for r in rows:
        print(f"{r['condition']:<34}{r['video_quality']:>5}{r['audio_quality']:>5}"
              f"{r['adaptive_weight_visual']:>7}{r['adaptive_weight_audio']:>7}"
              f"{r['adaptive_fused_emotion']:>10}{r['fixed_5050_emotion']:>9}"
              f"{('YES' if r['warning_banner'] else 'no'):>6}")
    print(f"\nSaved -> {OUT_PATH}")
    print("\nReading: as video quality falls (row 2) the adaptive weight shifts to")
    print("audio and the fused label follows the reliable channel; the fixed 50/50")
    print("baseline never adapts. The warning fires only when BOTH channels are poor.")


if __name__ == "__main__":
    run()
