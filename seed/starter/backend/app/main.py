"""FastAPI entry point for the starter site."""

from fastapi import FastAPI

from app.auth.routes import router as auth_router
from app.dashboard.routes import router as dashboard_router
from app.integrations.weather import router as weather_router

app = FastAPI(title="Starter site")
app.include_router(auth_router, prefix="/api/auth")
app.include_router(dashboard_router, prefix="/api/dashboard")
app.include_router(weather_router, prefix="/api/integrations")
