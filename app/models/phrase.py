from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB

Base = declarative_base()

class Phrase(Base):
    __tablename__ = "phrases"
    id = Column(Integer, primary_key=True)
    canonical_text = Column(Text, nullable=False)
    translations = Column(JSONB, nullable=False)  # {"ja": "...", "es": "..."}
    phonetic = Column(Text, nullable=True)
    context_category = Column(String(64), nullable=False, index=True)  # restaurant, transit, lodging, general
