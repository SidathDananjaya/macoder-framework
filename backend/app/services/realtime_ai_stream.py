# import cv2
# import asyncio
# import random
# import time

# # from backend.app.websocket.live_stream import manager


# class RealtimeAIStream:

#     def __init__(self):

#         self.camera = cv2.VideoCapture(0)

#         self.running = False

#     async def start_stream(self):

#         self.running = True

#         emotions = [
#             "happy",
#             "neutral",
#             "sad",
#             "angry",
#             "fear",
#             "disgust"
#         ]

#         stress_levels = [
#             "LOW",
#             "MEDIUM",
#             "HIGH"
#         ]

#         while self.running:

#             success, frame = self.camera.read()

#             if not success:
#                 continue

#             # -------------------------------------------------
#             # TEMPORARY MOCK AI PREDICTIONS
#             # -------------------------------------------------

#             emotion = random.choice(emotions)

#             temporal_emotion = random.choice(emotions)

#             stress = random.choice(stress_levels)

#             confidence = round(
#                 random.uniform(0.50, 0.99),
#                 2
#             )

#             # -------------------------------------------------
#             # LIVE DATA OBJECT
#             # -------------------------------------------------

#             live_data = {

#                 "emotion": emotion,

#                 "temporal_emotion":
#                     temporal_emotion,

#                 "stress_level":
#                     stress,

#                 "fusion_confidence":
#                     confidence
#             }

#             print("LIVE AI DATA:")
#             print(live_data)

#             # -------------------------------------------------
#             # SEND TO DASHBOARD
#             # -------------------------------------------------

#             await manager.broadcast(live_data)

#             await asyncio.sleep(1)

#     def stop_stream(self):

#         self.running = False

#         self.camera.release()