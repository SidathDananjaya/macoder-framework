from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect
)

from backend.app.websocket.live_stream import (
    manager
)

from backend.app.services.realtime_ai_engine import (
    process_frame
)

router = APIRouter()


@router.websocket("/ws/live")
async def websocket_endpoint(
    websocket: WebSocket
):

    await manager.connect(websocket)

    try:

        while True:

            frame_data = (
                await websocket.receive_text()
            )

            ai_result = await process_frame(
                frame_data
            )

            await websocket.send_json(
                ai_result
            )

    except WebSocketDisconnect:

        manager.disconnect(websocket)