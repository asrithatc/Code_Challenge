# src/app/models.py

from sqlalchemy import Column, Integer, String, Float, Date, UniqueConstraint
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class WeatherRecord(Base):
    __tablename__ = "weather_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    station_id = Column(String, nullable=False)
    date = Column(Date, nullable=False)

    max_temp = Column(Float)        # Celsius
    min_temp = Column(Float)        # Celsius
    precipitation = Column(Float)   # centimeters

    __table_args__ = (
        UniqueConstraint("station_id", "date", name="uix_station_date"),
    )


class WeatherStats(Base):
    __tablename__ = "weather_stats"

    id = Column(Integer, primary_key=True)
    station_id = Column(String, nullable=False)
    year = Column(Integer, nullable=False)

    avg_max_temp = Column(Float)
    avg_min_temp = Column(Float)
    total_precip_cm = Column(Float)

    __table_args__ = (
        UniqueConstraint("station_id", "year", name="uix_station_year"),
    )

