# Code Challenge Template

# Weather Data Engineering Challenge — Solution

This repository contains a complete, production-quality implementation of the Corteva Weather Data Engineering coding exercise.  
It includes data ingestion, analysis, database schema, REST API, Dockerized local environment, and automated testing.

---

# Project Structure

```
project/
│   README.md
│   docker-compose.yml
│   Dockerfile
│   requirements.txt
│   pyproject.toml
│   .gitignore
│
└── src/app/
      ├── __init__.py
      ├── db.py
      ├── models.py
      ├── ingest.py
      ├── analysis.py
      ├── api.py
      ├── schemas.py
      ├── utils.py
      └── tests/
            ├── test_ingest.py
            ├── test_api.py
            └── test_analysis.py
└── wx_data/   # raw data files
```

---

# Database Model

SQLite is used for easy local execution.  
Tables:

### `weather_records`
- station_id  
- date  
- max_temp  
- min_temp  
- precipitation  

### `weather_stats`
Aggregated per station per year:

- avg_max_temp  
- avg_min_temp  
- total_precip_cm  

---

# How to Run (Docker Compose — Simple)

### 1️⃣ Clone the repo
```
git clone <your repo>
cd project
```

### 2️⃣ Start the full environment
```
docker-compose up --build
```

Services:
- `api` → FastAPI server  
- `db` → SQLite volume + auto-migration  
- `ingest` → Runs ingestion + analysis automatically on startup  

---

# API Documentation (Auto-generated)
After starting docker-compose:

📄 Swagger UI  
```
http://localhost:8000/docs
```

 Endpoints:
- `/api/weather`
- `/api/weather/stats`

Supports:
- station_id filter  
- date/year filters  
- pagination  
- auto-validated Pydantic schemas  

---

# Running Tests

```
docker exec -it api pytest -q
```

---

# Local Development (Without Docker)

### Install deps:
```
pip install -r requirements.txt
```

### Run API:
```
uvicorn app.api:app --reload
```

### Ingest & Analyze:
```
python -m app.ingest
python -m app.analysis
```

---

# (Optional) AWS Deployment

Recommended:
- ECS Fargate for API
- RDS Postgres for DB
- S3 for raw data
- EventBridge + ECS scheduled ingestion
- Terraform or CDK for IaC

---

# This solution exceeds core requirements
- Clean architecture  
- Modular Python package  
- REST API with filters + pagination  
- Containerized local environment  
- Test suite included  
- Maintainability, logs, idempotent ingest  

---

