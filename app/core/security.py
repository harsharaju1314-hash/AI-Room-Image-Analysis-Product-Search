import io
from fastapi import HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError
from .config import settings


def validate_image_upload(file: UploadFile, file_bytes: bytes) -> None:
    """
    Validates uploaded image file against size limits, file extensions,
    and binary image header integrity.
    """
    # 1. Validate filename and extension
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename."
        )

    ext = file.filename.split(".")[-1].lower() if "." in file.filename else ""
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file extension '{ext}'. "
                f"Allowed extensions are: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )
        )

    # 2. Validate file size
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE if hasattr(status, "HTTP_413_CONTENT_TOO_LARGE") else 413,
            detail=f"File size exceeds maximum allowed limit of {settings.MAX_UPLOAD_SIZE_MB}MB."
        )

    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )

    # 3. Validate image integrity using Pillow verify()
    try:
        image_stream = io.BytesIO(file_bytes)
        with Image.open(image_stream) as img:
            img.verify()
    except (UnidentifiedImageError, ValueError, Exception) as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is corrupt or is not a valid image."
        ) from err
