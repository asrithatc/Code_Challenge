# Problem 2 — Ingestion

## Approach
I implemented an ingestion pipeline that:
1. Iterates through all files in the `wx_data/` directory.
2. Parses each line into:
   - date
   - max_temp (converted from tenths of degrees Celsius)
   - min_temp
   - precipitation (converted from tenths of millimeters → centimeters)
3. Normalizes missing values (-9999 → NULL)
4. Inserts rows into SQLite using SQLAlchemy.

## Idempotency
Before inserting each record, the ingestion code checks:

```python
db.query(WeatherRecord).filter_by(station_id=station, date=date).first()


If it exists → skip
If not → insert
This ensures re-running ingestion NEVER duplicates records.
Logging
The ingestion process logs:
Start time
End time
Total number of inserted rows
Summary
The ingestion step loaded ~1.7 million raw weather records successfully.

