"""Proxy to the external weather API with a 10-minute cache."""

import os
import time

import httpx
from fastapi import APIRouter, HTTPException

router = APIRouter()
_CACHE: dict[str, tuple[float, dict]] = {}
TTL_SECONDS = 600


@router.get("/weather")
def weather(city: str) -> dict:
    hit = _CACHE.get(city)
    if hit and time.time() - hit[0] < TTL_SECONDS:
        return hit[1]
    try:
        resp = httpx.get(
            "https://api.weather.example.com/v1/current",
            params={"q": city, "key": os.environ.get("WEATHER_API_KEY", "")},
            timeout=5,
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Weather API failed") from exc
    data = resp.json()
    out = {"city": city, "temp_c": data["temp_c"], "condition": data["condition"]}
    _CACHE[city] = (time.time(), out)
    return out
