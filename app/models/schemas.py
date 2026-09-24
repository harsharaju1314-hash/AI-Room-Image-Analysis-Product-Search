from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class ProductBase(BaseModel):
    name: str = Field(..., json_schema_extra={"example": "Modern Fabric Sofa"})
    category: str = Field(..., json_schema_extra={"example": "sofa"})
    room_type: str = Field(..., json_schema_extra={"example": "living_room"})
    price: float = Field(..., json_schema_extra={"example": 799.99})
    description: str = Field(..., json_schema_extra={"example": "Comfortable 3-seater contemporary sofa in neutral grey."})
    image_url: Optional[str] = Field(None, json_schema_extra={"example": "/static/products/sofa_01.jpg"})


class ProductCreate(ProductBase):
    embedding: List[float]


class ProductResponse(ProductBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    similarity_score: Optional[float] = Field(None, description="Cosine similarity score (0.0 to 1.0)")
    created_at: Optional[datetime] = None


class ImageMetrics(BaseModel):
    width: int
    height: int
    channels: int
    shannon_entropy: float = Field(..., description="Visual complexity / entropy calculated via scikit-image")
    contrast_metric: float = Field(..., description="GLCM contrast measurement calculated via scikit-image")


class RoomAnalysisResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    room_type: str = Field(..., json_schema_extra={"example": "living_room"})
    confidence: float = Field(..., json_schema_extra={"example": 0.92})
    class_probabilities: dict[str, float]
    image_metrics: ImageMetrics
    similar_products: List[ProductResponse]
    created_at: datetime


class HealthResponse(BaseModel):
    status: str = "healthy"
    app_name: str
    environment: str
    model_loaded: bool
    database_connected: bool
    timestamp: datetime
