#!/usr/bin/env python3
"""
Simple startup script for the Captcha Solve API
Run this from the project root directory
"""

import uvicorn
import os
import sys

def main():
    """Start the FastAPI server"""
    
    # Add current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Check if model file exists
    model_path = "model/model.pth"
    if not os.path.exists(model_path):
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
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,  # Disable reload to avoid subprocess issues
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
