# src/app/db.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models import Base

engine = create_engine("sqlite:///weather.db", echo=False, future=True)
SessionLocal = sessionmaker(bind=engine)

Base.metadata.create_all(engine)

