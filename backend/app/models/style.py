"""
Database models for Style storage
"""
from sqlalchemy import Column, String, Float, DateTime, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid

Base = declarative_base()


class Style(Base):
    """Style model for storing makeup filter styles"""
    __tablename__ = "styles"
    
    # Primary key
    style_id = Column(String(64), primary_key=True, default=lambda: f"style_{uuid.uuid4().hex[:8]}")
    
    # Metadata
    name = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    
    # Style parameters (stored as JSON)
    style_parameters = Column(JSON, nullable=False)
    
    # Asset URLs/paths
    lut_lips_url = Column(String(512), nullable=True)
    lut_eyes_url = Column(String(512), nullable=True)
    lut_skin_url = Column(String(512), nullable=True)
    lut_eyebrows_url = Column(String(512), nullable=True)
    lut_cheeks_url = Column(String(512), nullable=True)
    
    shader_fragment_url = Column(String(512), nullable=True)
    shader_vertex_url = Column(String(512), nullable=True)
    
    thumbnail_url = Column(String(512), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Statistics
    usage_count = Column(Float, default=0.0, nullable=False)
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'style_id': self.style_id,
            'name': self.name,
            'description': self.description,
            'style_parameters': self.style_parameters,
            'lut_lips_url': self.lut_lips_url,
            'lut_eyes_url': self.lut_eyes_url,
            'lut_skin_url': self.lut_skin_url,
            'lut_eyebrows_url': self.lut_eyebrows_url,
            'lut_cheeks_url': self.lut_cheeks_url,
            'shader_fragment_url': self.shader_fragment_url,
            'shader_vertex_url': self.shader_vertex_url,
            'thumbnail_url': self.thumbnail_url,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'usage_count': self.usage_count
        }

