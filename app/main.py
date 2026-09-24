import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.db.session import init_db
from app.services.vision_model import get_vision_service
from app.api.endpoints import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown event handling.
    Initializes database tables and pre-warms the PyTorch model.
    """
    logger.info("Initializing database...")
    init_db()

    logger.info("Warming up PyTorch computer vision model...")
    try:
        get_vision_service()
        logger.info("Computer vision model successfully loaded into memory.")
    except Exception as e:
        logger.error(f"Error loading vision model: {e}")

    yield

    logger.info("Shutting down application...")


app = FastAPI(
    title=settings.APP_NAME,
    description="AI Room Image Analysis and Product Search service using PyTorch, OpenCV, scikit-image, and pgvector.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please try again later."}
    )

# Include API routes
app.include_router(router)


@app.get("/", tags=["Root"])
def root():
    return {
        "message": "AI Room Image Analysis & Product Search API",
        "documentation": "/docs",
        "health_check": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
