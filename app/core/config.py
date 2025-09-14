"""
Configuration settings for the Captcha Solve API
"""

import os
from typing import Optional

class Config:
    """Configuration class for the API"""
    
    # API Settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_TITLE: str = "Captcha Solve API"
    API_VERSION: str = "1.0.0"
    
    # Model Settings
    MODEL_PATH: str = os.getenv("MODEL_PATH", "model/model.pth")
    MODEL_DEVICE: str = os.getenv("MODEL_DEVICE", "auto")  # auto, cpu, cuda
    
    # Database Settings
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", "captcha_api.db")
    DATABASE_TIMEOUT: float = float(os.getenv("DATABASE_TIMEOUT", "30.0"))
    DATABASE_RETRY_ATTEMPTS: int = int(os.getenv("DATABASE_RETRY_ATTEMPTS", "3"))
    
    # CORS Settings
    CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # File Upload Settings
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB
    ALLOWED_EXTENSIONS: set = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"}
    
    # Logging Settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def get_model_device(cls) -> str:
        """Get the appropriate device for the model"""
        if cls.MODEL_DEVICE == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return cls.MODEL_DEVICE
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate the configuration"""
        # Check if model file exists
        if not os.path.exists(cls.MODEL_PATH):
            print(f"Warning: Model file not found at {cls.MODEL_PATH}")
            return False
        
        # Check if port is valid
        if not (1 <= cls.API_PORT <= 65535):
            print(f"Error: Invalid port number {cls.API_PORT}")
            return False
        
        return True

# Development configuration
class DevelopmentConfig(Config):
    """Development configuration"""
    LOG_LEVEL = "DEBUG"
    CORS_ORIGINS = ["*"]

# Production configuration
class ProductionConfig(Config):
    """Production configuration"""
    LOG_LEVEL = "WARNING"
    CORS_ORIGINS = ["https://yourdomain.com"]  # Update with your domain

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config(env: Optional[str] = None) -> Config:
    """Get configuration based on environment"""
    if env is None:
        env = os.getenv("FLASK_ENV", "default")
    
    return config.get(env, config['default'])
