"""
API endpoints for the Captcha Solve API
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from datetime import datetime
import uuid
from typing import Optional
from io import BytesIO
from PIL import Image
import logging
import os
import pathlib

from app.models.predict import CaptchaPredictor
from app.core.database import get_db_manager

logger = logging.getLogger(__name__)

# Create router for captcha-related endpoints
captcha_router = APIRouter(prefix="/api/v1", tags=["captcha"])

# Global predictor instance (will be set from main.py)
predictor = None

# Backup directory for storing captcha images
BACKUP_DIR = pathlib.Path("captcha_backup")
BACKUP_DIR.mkdir(exist_ok=True)

def set_predictor(pred: CaptchaPredictor):
    """Set the global predictor instance"""
    global predictor
    predictor = pred

def save_image_to_backup(image_data: bytes, request_id: str, captcha_hash: str) -> str:
    """
    Save image to backup directory with unique filename
    
    Args:
        image_data: The image binary data
        request_id: Unique request identifier
        captcha_hash: Captcha hash for additional uniqueness
    
    Returns:
        str: The filepath where the image was saved
    """
    try:
        # Create unique filename using request_id and captcha_hash
        filename = f"{request_id}_{captcha_hash}.png"
        filepath = BACKUP_DIR / filename
        
        # Save image data to file
        with open(filepath, 'wb') as f:
            f.write(image_data)
        
        logger.info(f"Saved captcha image to backup: {filepath}")
        return str(filepath)
        
    except Exception as e:
        logger.error(f"Failed to save image to backup: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save image backup: {str(e)}")

@captcha_router.post("/solve")
async def solve_captcha(
    captcha_hash: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Solve a captcha image
    
    Args:
        captcha_hash: Hash identifier for the captcha
        image: The captcha image file
    
    Returns:
        JSON response with predicted answer and confidence
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Captcha predictor not initialized")
    
    try:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Read and validate image
        image_data = await image.read()
        
        # Validate image format
        try:
            validation_image = Image.open(BytesIO(image_data))
            validation_image.verify()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")
        
        image_io = BytesIO(image_data)
        image_io.seek(0)
        pil_image = Image.open(image_io)
        
        # Save image to backup directory
        image_filepath = save_image_to_backup(image_data, request_id, captcha_hash)
        
        predicted_answer, confidence = predictor.predict_single(pil_image)
      
        db_manager = get_db_manager()
        success = db_manager.insert_captcha_request(
            request_id, captcha_hash, predicted_answer, confidence, image_filepath
        )
        
        if not success:
            logger.error(f"Failed to store captcha request: {request_id}")
            raise HTTPException(status_code=500, detail="Failed to store request in database")
        
        return {
            "request_id": request_id,
            "captcha_hash": captcha_hash,
            "predicted_answer": predicted_answer,
            "confidence": round(confidence, 4),
            "image_backup_path": image_filepath,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing captcha: {str(e)}")

@captcha_router.post("/feedback")
async def submit_feedback(
    request_id: str = Form(...),
    is_correct: bool = Form(...),
    actual_answer: Optional[str] = Form(None)
):
    """
    Submit feedback on a captcha prediction
    
    Args:
        request_id: The request ID from the solve endpoint
        is_correct: Whether the prediction was correct
        actual_answer: The actual answer (optional)
    
    Returns:
        JSON response confirming feedback submission
    """
    try:
        # Use thread-safe database manager
        db_manager = get_db_manager()
        success = db_manager.insert_feedback(request_id, is_correct, actual_answer)
        
        if not success:
            # Check if request exists
            request_data = db_manager.get_captcha_request(request_id)
            if not request_data:
                raise HTTPException(status_code=404, detail="Request ID not found")
            else:
                raise HTTPException(status_code=500, detail="Failed to store feedback")
        
        return {
            "message": "Feedback submitted successfully",
            "request_id": request_id,
            "is_correct": is_correct,
            "actual_answer": actual_answer,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting feedback: {str(e)}")

@captcha_router.get("/stats")
async def get_stats():
    """
    Get API usage statistics
    
    Returns:
        JSON response with usage statistics
    """
    try:
        # Use thread-safe database manager
        db_manager = get_db_manager()
        stats = db_manager.get_stats()
        stats["timestamp"] = datetime.now().isoformat()
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving stats: {str(e)}")

# Health check router
health_router = APIRouter(tags=["health"])

@health_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "predictor_loaded": predictor is not None,
        "timestamp": datetime.now().isoformat()
    }
