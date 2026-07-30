from ai_engine.fusion.multimodal_cognitive_fusion import (
    MultimodalCognitiveFusion
)

fusion_engine = MultimodalCognitiveFusion()


async def fuse_results(visual_result, audio_result):

    try:

        fusion = fusion_engine.process(

            visual_emotion=visual_result.get("emotion", "neutral"),
            visual_emotion_confidence=visual_result.get(
                "visual_emotion_confidence", 0.0
            ),
            audio_emotion=audio_result.get("audio_emotion", "neutral"),
            audio_emotion_confidence=audio_result.get(
                "audio_confidence", 0.0
            ),

            visual_stress=visual_result.get("visual_stress", "LOW"),
            visual_stress_confidence=1.0,
            audio_stress=audio_result.get("audio_stress", "LOW"),
            audio_stress_confidence=audio_result.get(
                "audio_stress_confidence", 0.0
            ),

            blink_rate=visual_result.get("blink_count", 0),
            head_movement=visual_result.get("movement_score", 0),
            gaze_stability=visual_result.get("gaze_stability", 1.0),
            gaze_shift_frequency=visual_result.get(
                "gaze_shift_frequency", 0
            ),

            video_quality=visual_result.get("video_quality", 1.0),
            audio_quality=audio_result.get("audio_quality", 1.0),
        )

        fusion["stress_level"] = fusion["fused_stress"]

        fusion["fused_emotion"] = fusion.pop("final_emotion", None)

        return fusion

    except Exception as e:

        print("FUSION ERROR:", e)

        return {}
