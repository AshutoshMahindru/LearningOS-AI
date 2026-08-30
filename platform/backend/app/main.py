from fastapi import FastAPI
from app.api import routes

app = FastAPI(
    title="LearningOS V3 Generic Application API",
    version="3.0.0",
    description="Strictly generic REST API powering the local-first LearningOS V3 platform."
)

app.include_router(routes.router, prefix="/api/v1")
