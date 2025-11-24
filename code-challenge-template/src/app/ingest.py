# src/app/ingest.py

import os
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from app.models import WeatherRecord
from app.db import SessionLocal
from app.utils import parse_wx_line

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_weather_data(wx_dir: str):
    print("INGEST SCRIPT IS RUNNING")
    db: Session = SessionLocal()
    start = datetime.now()
    logger.info(f"[INGEST] Started ingestion at {start}")

    records_inserted = 0

    for filename in os.listdir(wx_dir):
        station_id = filename.replace(".txt", "")
        filepath = os.path.join(wx_dir, filename)

        with open(filepath, "r") as f:
            for line in f:
                date, max_t, min_t, prec = parse_wx_line(line)

                exists = (
                    db.query(WeatherRecord)
                    .filter_by(station_id=station_id, date=date)
                    .first()
                )
                if exists:
                    continue

                rec = WeatherRecord(
                    station_id=station_id,
                    date=date,
                    max_temp=max_t,
                    min_temp=min_t,
                    precipitation=prec,
                )
                db.add(rec)
                records_inserted += 1

    db.commit()
    end = datetime.now()
    logger.info(f"[INGEST] Completed ingestion at {end}")
    logger.info(f"[INGEST] Total new records inserted: {records_inserted}")
    db.close()

if __name__ == "__main__":
    ingest_weather_data("wx_data")

