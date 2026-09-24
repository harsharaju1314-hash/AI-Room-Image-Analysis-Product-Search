from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.core.config import settings
from app.core.security import validate_image_upload
from app.db.session import get_db
from app.models.entities import Product, RoomAnalysis
from app.models.schemas import (
    HealthResponse,
    ImageMetrics,
    ProductResponse,
    RoomAnalysisResponse,
)
from app.services.image_processor import ImageProcessor
from app.services.search_service import ProductSearchService
from app.services.vision_model import get_vision_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check(db: Session = Depends(get_db)):
    """
    Health check endpoint returning system status, model availability, and DB connection.
    """
    vision_service = get_vision_service()
    model_loaded = vision_service.model is not None

    db_connected = False
    try:
        db.execute(text("SELECT 1;"))
        db_connected = True
    except Exception:
        db_connected = False

    return HealthResponse(
        status="healthy" if (model_loaded and db_connected) else "degraded",
        app_name=settings.APP_NAME,
        environment=settings.APP_ENV,
        model_loaded=model_loaded,
        database_connected=db_connected,
        timestamp=datetime.now(timezone.utc)
    )


@router.post("/analyze", response_model=RoomAnalysisResponse, status_code=status.HTTP_201_CREATED, tags=["Analysis"])
async def analyze_room_image(
    file: UploadFile = File(..., description="Room / Interior image file (JPG, PNG, WebP)"),
    top_k: int = Query(3, ge=1, le=10, description="Number of similar products to retrieve"),
    filter_by_room: bool = Query(True, description="Filter product search results to the classified room type"),
    db: Session = Depends(get_db)
):
    """
    Analyzes an uploaded room image:
    1. Validates file security (extension, size, image header).
    2. Extracts image complexity & contrast metrics via scikit-image & OpenCV.
    3. Runs PyTorch ResNet-18 forward pass to classify room category and extract 512-dim embedding.
    4. Searches product catalog for visually similar items using vector cosine similarity (pgvector).
    5. Stores the room analysis record and returns comprehensive results.
    """
    # 1. Read file bytes safely
    file_bytes = await file.read()

    # 2. Validate upload security
    validate_image_upload(file, file_bytes)

    try:
        # 3. Process image with Pillow and scikit-image
        pil_image = ImageProcessor.load_image_from_bytes(file_bytes)
        metrics_dict = ImageProcessor.calculate_image_metrics(pil_image)

        # 4. ML Model Inference (PyTorch ResNet-18)
        vision_service = get_vision_service()
        embedding, room_type, confidence, class_probs = vision_service.process_image(pil_image)

        # 5. Vector Similarity Search against Product Catalog
        room_filter = room_type if filter_by_room else None
        similar_products = ProductSearchService.search_similar_products(
            db=db,
            query_embedding=embedding,
            room_type_filter=room_filter,
            top_k=top_k
        )

        matched_ids = [p.id for p in similar_products]

        # 6. Persist Room Analysis to PostgreSQL
        analysis_record = RoomAnalysis(
            filename=file.filename,
            room_type=room_type,
            confidence=confidence,
            class_probabilities=class_probs,
            image_metrics=metrics_dict,
            matched_product_ids=matched_ids,
            embedding=embedding,
            created_at=datetime.now(timezone.utc)
        )
        db.add(analysis_record)
        db.commit()
        db.refresh(analysis_record)

        return RoomAnalysisResponse(
            id=analysis_record.id,
            filename=analysis_record.filename,
            room_type=analysis_record.room_type,
            confidence=analysis_record.confidence,
            class_probabilities=analysis_record.class_probabilities,
            image_metrics=ImageMetrics(**analysis_record.image_metrics),
            similar_products=similar_products,
            created_at=analysis_record.created_at
        )

    except HTTPException:
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the image: {str(exc)}"
        )


@router.get("/analysis/{analysis_id}", response_model=RoomAnalysisResponse, tags=["Analysis"])
def get_analysis_by_id(analysis_id: int, db: Session = Depends(get_db)):
    """
    Retrieves a past room analysis record and associated recommended products by ID.
    """
    record = db.query(RoomAnalysis).filter(RoomAnalysis.id == analysis_id).first()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis with ID {analysis_id} not found."
        )

    # Fetch matched products from IDs
    matched_ids = record.matched_product_ids or []
    products = db.query(Product).filter(Product.id.in_(matched_ids)).all() if matched_ids else []
    
    product_responses = [
        ProductResponse(
            id=p.id,
            name=p.name,
            category=p.category,
            room_type=p.room_type,
            price=float(p.price),
            description=p.description or "",
            image_url=p.image_url,
            similarity_score=None,
            created_at=p.created_at
        )
        for p in products
    ]

    return RoomAnalysisResponse(
        id=record.id,
        filename=record.filename,
        room_type=record.room_type,
        confidence=record.confidence,
        class_probabilities=record.class_probabilities,
        image_metrics=ImageMetrics(**record.image_metrics),
        similar_products=product_responses,
        created_at=record.created_at
    )


@router.get("/products/{product_id}", response_model=ProductResponse, tags=["Products"])
def get_product_by_id(product_id: int, db: Session = Depends(get_db)):
    """
    Retrieves details for a specific catalog product.
    """
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product with ID {product_id} not found."
        )

    return ProductResponse(
        id=product.id,
        name=product.name,
        category=product.category,
        room_type=product.room_type,
        price=float(product.price),
        description=product.description or "",
        image_url=product.image_url,
        similarity_score=None,
        created_at=product.created_at
    )
