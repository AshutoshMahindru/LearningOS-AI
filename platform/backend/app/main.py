from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes
from app.core.config import attempt_storage_init, ensure_data_layout, get_settings
from app.core.errors import register_error_handlers
from app.core.security import ensure_auth_token
from app.core.version import PLATFORM_VERSION

LOCAL_FRONTEND_ORIGINS = (
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://127.0.0.1:4173",
    "http://localhost:4173",
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    ensure_data_layout(settings)
    ensure_auth_token()
    attempt_storage_init(settings)
    yield


app = FastAPI(
    title="LearningOS V3 Generic Application API",
    version=PLATFORM_VERSION,
    description="Strictly generic REST API powering the local-first LearningOS V3 platform.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(LOCAL_FRONTEND_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_error_handlers(app)
app.include_router(routes.router, prefix="/api/v1")
