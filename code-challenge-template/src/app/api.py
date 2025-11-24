# src/app/api.py

from fastapi import FastAPI, Query
from fastapi import Depends
from sqlalchemy.orm import Session
from app.db import SessionLocal
from app.models import WeatherRecord, WeatherStats
from app.schemas import WeatherRecordSchema, WeatherStatsSchema

app = FastAPI(title="Weather API", version="1.0")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/api/weather", response_model=list[WeatherRecordSchema])
def get_weather(
    station_id: str | None = None,
    date: str | None = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):

    q = db.query(WeatherRecord)
    if station_id:
        q = q.filter(WeatherRecord.station_id == station_id)
    if date:
        q = q.filter(WeatherRecord.date == date)

    return q.offset(skip).limit(limit).all()


@app.get("/api/weather/stats", response_model=list[WeatherStatsSchema])
def get_weather_stats(
    station_id: str | None = None,
    year: int | None = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    q = db.query(WeatherStats)
    if station_id:
        q = q.filter(WeatherStats.station_id == station_id)
    if year:
        q = q.filter(WeatherStats.year == year)

    return q.offset(skip).limit(limit).all()

