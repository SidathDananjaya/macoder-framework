# from fastapi import APIRouter
# import asyncio

# from backend.app.services.realtime_ai_stream import RealtimeAIStream

# router = APIRouter()

# stream_engine = RealtimeAIStream()


# @router.get("/start-stream")

# async def start_stream():

#     asyncio.create_task(
#         stream_engine.start_stream()
#     )

#     return {
#         "status": "AI stream started"
#     }


# @router.get("/stop-stream")

# async def stop_stream():

#     stream_engine.stop_stream()

#     return {
#         "status": "AI stream stopped"
#     }