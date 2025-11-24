# src/app/schemas.py

from pydantic import BaseModel
from datetime import date

class WeatherRecordSchema(BaseModel):
    station_id: str
    date: date
    max_temp: float | None
    min_temp: float | None
    precipitation: float | None

    model_config = {
        "from_attributes": True
    }


class WeatherStatsSchema(BaseModel):
    station_id: str
    year: int
    avg_max_temp: float | None
    avg_min_temp: float | None
    total_precip_cm: float | None

    model_config = {
        "from_attributes": True
    }

