from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes.stress_routes import (
    router as stress_router
)

from backend.app.api.routes.session_routes import (
    router as session_router
)

from backend.app.api.routes.websocket_stream import (
    router as websocket_router
)

from backend.app.api.routes.session_export import (
    router as export_router
)

app = FastAPI(
    title="MaCoDeR API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    stress_router,
    prefix="/api"
)

app.include_router(
    session_router,
    prefix="/api"
)

app.include_router(
    websocket_router
)

app.include_router(
    export_router,
    prefix="/api"
)

@app.get("/")
def root():

    return {
        "message": "MaCoDeR API Running"
    }