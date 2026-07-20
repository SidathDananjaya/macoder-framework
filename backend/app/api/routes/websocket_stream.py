from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect
)

from backend.app.websocket.live_stream import (
    manager
)

from backend.app.services.realtime_ai_engine import (
    process_frame,
    reset_engine
)

from backend.app.services.realtime_audio_engine import (
    process_audio,
    reset_audio
)

from backend.app.services.realtime_fusion_engine import (
    fuse_results
)

from backend.app.services.session_logger_service import (
    log_session,
    reset_session,
    save_session_json
)

from backend.app.services.session_memory_service import (
    update_memory,
    reset_memory
)

router = APIRouter()


@router.websocket("/ws/live")
async def websocket_endpoint(
    websocket: WebSocket
):

    await manager.connect(websocket)

    # Fresh vision state (blink total, emotion cache, movement/gaze) per session.
    reset_engine()

    # Fresh rolling-memory windows for each session.
    await reset_memory()

    # Fresh rolling audio window for each session.
    reset_audio()

    # Phase 7 - begin a new frame-by-frame session recording.
    await reset_session()

    try:

        while True:

            payload = (
                await websocket.receive_json()
            )

            frame_data = payload.get(
                "frame"
            )

            audio_data = payload.get(
                "audio"
            )

            # Browser's native capture rate (44100/48000); default to the model rate if not sent.
            sample_rate = payload.get(
                "sample_rate",
                22050
            )

            visual_result = (
                await process_frame(
                    frame_data
                )
            )

            # Buffer a rolling window, resample to 22050 Hz and infer (throttled); always returns a result.
            audio_result = (
                await process_audio(
                    audio_data,
                    sample_rate
                )
            )

            fusion_result = (
                await fuse_results(
                    visual_result,
                    audio_result
                )
            )

            final_result = {

                **visual_result,

                **audio_result,

                **fusion_result
            }

            await log_session(
                final_result
            )

            # Rolling 5s/10s/30s memory, sent to the client but kept out of the log so the CSV stays flat.
            session_memory = (
                await update_memory(final_result)
            )

            await websocket.send_json({
                **final_result,
                "session_memory": session_memory
            })

    except WebSocketDisconnect:

        manager.disconnect(
            websocket
        )

        # Save the finished recording as JSON so the report generator always has it, even without an explicit export.
        saved = await save_session_json()

        if saved:
            print("SESSION RECORDING SAVED:", saved)