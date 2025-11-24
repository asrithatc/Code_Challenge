# src/app/analysis.py

from sqlalchemy import func
from sqlalchemy.orm import Session
from app.models import WeatherRecord, WeatherStats
from app.db import SessionLocal

def compute_weather_stats():
    db: Session = SessionLocal()

    results = (
        db.query(
            WeatherRecord.station_id,
            func.extract("year", WeatherRecord.date).label("year"),
            func.avg(WeatherRecord.max_temp),
            func.avg(WeatherRecord.min_temp),
            func.sum(WeatherRecord.precipitation),
        )
        .group_by("station_id", "year")
        .all()
    )

    for station_id, year, avg_max, avg_min, total_prec in results:
        stat = db.query(WeatherStats).filter_by(station_id=station_id, year=int(year)).first()
        if not stat:
            stat = WeatherStats(
                station_id=station_id,
                year=int(year),
                avg_max_temp=avg_max,
                avg_min_temp=avg_min,
                total_precip_cm=total_prec,
            )
            db.add(stat)

    db.commit()
    db.close()

