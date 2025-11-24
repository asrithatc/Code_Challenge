# Problem 1 — Data Modeling

## Database Choice
I used **SQLite** for this exercise because:
- It requires no external infrastructure
- It is easy to containerize and run locally
- SQLAlchemy supports it cleanly with no platform overhead

For production, I would use **PostgreSQL** on AWS RDS.

## Data Model
I modeled the weather data using SQLAlchemy ORM with two tables:

### `weather_records`
Stores raw daily measurements.

Fields:
- station_id (string)
- date (date)
- max_temp (float)
- min_temp (float)
- precipitation (float)

A unique constraint on `(station_id, date)` ensures ingestion is idempotent.

### `weather_stats`
Stores aggregated yearly station statistics.

Fields:
- station_id (string)
- year (int)
- avg_max_temp (float)
- avg_min_temp (float)
- total_precip_cm (float)

This table also has a unique constraint `(station_id, year)`.

The schema is normalized and optimized for analytical reads.

