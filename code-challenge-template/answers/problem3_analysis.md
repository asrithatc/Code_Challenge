
---

# 📘 **answers/problem3_analysis.md**

```markdown
# Problem 3 — Data Analysis

## Goal
For each station and each year, compute:
- Average max temperature
- Average min temperature
- Total precipitation

Missing values are excluded from averages.

## Implementation
A SQLAlchemy query groups raw data:

```python
db.query(
    WeatherRecord.station_id,
    extract("year", WeatherRecord.date),
    avg(WeatherRecord.max_temp),
    avg(WeatherRecord.min_temp),
    sum(WeatherRecord.precipitation)
)


Results are written to the weather_stats table.
Null Handling
If a station-year has insufficient data:
Averages become NULL automatically
Total precipitation may still exist
Summary
The analysis step executed successfully and populated the weather_stats table with yearly aggregates.

