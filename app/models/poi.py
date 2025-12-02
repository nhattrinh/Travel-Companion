from sqlalchemy import Column, Integer, String, Float, Text
from app.core.db import Base


class POI(Base):
    __tablename__ = "pois"
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    category = Column(String(64), nullable=False)
    latitude = Column(Float, nullable=False, index=True)
    longitude = Column(Float, nullable=False, index=True)
    etiquette_notes = Column(Text, nullable=True)
