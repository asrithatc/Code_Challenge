
---

# 📘 **answers/problem4_rest_api.md**

```markdown
# Problem 4 — REST API

## Framework Choice
I used **FastAPI** because:
- Automatic OpenAPI/Swagger documentation
- Native Pydantic validation
- Excellent performance and clean routing

## Endpoints

### `/api/weather`
Returns raw daily weather data.  
Supports:
- station_id filter
- date filter
- pagination (`skip`, `limit`)

### `/api/weather/stats`
Returns aggregated statistics.  
Supports:
- station_id filter
- year filter
- pagination

## Pagination
Implemented via SQLAlchemy query:
```python
q.offset(skip).limit(limit)



Summary
The API exposes all required data cleanly, validates responses using Pydantic, and supports filters + pagination.

