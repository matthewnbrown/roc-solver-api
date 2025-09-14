#!/usr/bin/env python3
"""
Startup script for the Captcha Solve API
"""

import uvicorn
import os
import sys
from pathlib import Path

def main():
    """Start the FastAPI server"""
    
    # Add the project root to Python path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # Check if model file exists
    model_path = project_root / "model" / "2025_09_12best_model.pth"
    if not model_path.exists():
        print(f"Warning: Model file not found at {model_path}")
        print("The API will start but captcha solving will not work.")
        print("Please ensure the model file is in the correct location.")
        print()
    
    # Start the server
    print("Starting Captcha Solve API server...")
    print("API will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    print("Endpoints available at: http://localhost:8000/api/v1/")
    print("Press Ctrl+C to stop the server")
    print()
    
    try:
        uvicorn.run(
            "app.main:app",  # Updated to use new app structure
            host="0.0.0.0",
            port=8000,
            reload=True,  # Enable auto-reload for development
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()