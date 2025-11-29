from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Float, Text

Base = declarative_base()

class POI(Base):
    __tablename__ = "pois"
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    category = Column(String(64), nullable=False)
    latitude = Column(Float, nullable=False, index=True)
    longitude = Column(Float, nullable=False, index=True)
    etiquette_notes = Column(Text, nullable=True)
