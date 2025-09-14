from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import logging

from app.models.predict import CaptchaPredictor
from app.core.database import init_database_manager
from app.core.config import Config
from app.api.endpoints import captcha_router, health_router, set_predictor

app = FastAPI(title=Config.API_TITLE, version=Config.API_VERSION)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Include routers
app.include_router(captcha_router)
app.include_router(health_router)

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    # Initialize database manager with configuration
    init_database_manager(
        db_path=Config.DATABASE_PATH,
        timeout=Config.DATABASE_TIMEOUT,
        retry_attempts=Config.DATABASE_RETRY_ATTEMPTS
    )
    logger.info("Database manager initialized successfully")
    
    # Initialize the predictor
    model_path = Config.MODEL_PATH
    if os.path.exists(model_path):
        predictor = CaptchaPredictor(model_path, device=Config.get_model_device())
        set_predictor(predictor)
        logger.info("Captcha predictor initialized successfully")
    else:
        logger.warning(f"Model file not found at {model_path}")
        logger.warning("API will be available but captcha solving will not work")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": Config.API_TITLE,
        "version": Config.API_VERSION,
        "endpoints": {
            "solve": "/api/v1/solve - POST captcha image and hash",
            "feedback": "/api/v1/feedback - POST feedback on prediction accuracy",
            "stats": "/api/v1/stats - GET API usage statistics",
            "health": "/health - GET health status"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=Config.API_HOST, port=Config.API_PORT)
