#!/usr/bin/env python3
"""
Example client for the Captcha Solve API
"""

import requests
import json
import os
from typing import Optional, Dict, Any

class CaptchaAPIClient:
    """Client for interacting with the Captcha Solve API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the API is healthy"""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def solve_captcha(self, image_path: str, captcha_hash: str) -> Dict[str, Any]:
        """
        Solve a captcha image
        
        Args:
            image_path: Path to the captcha image file
            captcha_hash: Unique identifier for the captcha
            
        Returns:
            Dictionary containing prediction results
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {'captcha_hash': captcha_hash}
            
            response = requests.post(f"{self.base_url}/api/v1/solve", files=files, data=data)
            response.raise_for_status()
            
        return response.json()
    
    def submit_feedback(self, request_id: str, is_correct: bool, actual_answer: Optional[str] = None) -> Dict[str, Any]:
        """
        Submit feedback on a prediction
        
        Args:
            request_id: The request ID from the solve endpoint
            is_correct: Whether the prediction was correct
            actual_answer: The actual answer (optional)
            
        Returns:
            Dictionary confirming feedback submission
        """
        data = {
            'request_id': request_id,
            'is_correct': is_correct
        }
        
        if actual_answer is not None:
            data['actual_answer'] = actual_answer
        
        response = requests.post(f"{self.base_url}/api/v1/feedback", data=data)
        response.raise_for_status()
        
        return response.json()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get API usage statistics"""
        response = requests.get(f"{self.base_url}/api/v1/stats")
        response.raise_for_status()
        return response.json()

def main():
    """Example usage of the CaptchaAPIClient"""
    
    # Initialize client
    client = CaptchaAPIClient()
    
    try:
        # Check API health
        print("Checking API health...")
        health = client.health_check()
        print(f"API Status: {health['status']}")
        print(f"Predictor Loaded: {health['predictor_loaded']}")
        print()
        
        # Example: Solve a captcha (you would replace this with a real image)
        image_path = "example_captcha.png"
        captcha_hash = "example_hash_123"
        
        if os.path.exists(image_path):
            print(f"Solving captcha: {image_path}")
            result = client.solve_captcha(image_path, captcha_hash)
            
            print(f"Predicted Answer: {result['predicted_answer']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Request ID: {result['request_id']}")
            print()
            
            # Example: Submit feedback (simulate user feedback)
            print("Submitting feedback...")
            feedback = client.submit_feedback(
                request_id=result['request_id'],
                is_correct=True,  # Assume prediction was correct
                actual_answer=result['predicted_answer']
            )
            print(f"Feedback submitted: {feedback['message']}")
            print()
        
        else:
            print(f"Example image not found: {image_path}")
            print("Create a captcha image file to test the solve functionality")
            print()
        
        # Get statistics
        print("Getting API statistics...")
        stats = client.get_stats()
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Accuracy: {stats['accuracy_percentage']:.2f}%")
        print(f"Average Confidence: {stats['average_confidence']:.4f}")
        print(f"Recent Requests (24h): {stats['recent_requests_24h']}")
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server.")
        print("Make sure the server is running on http://localhost:8000")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        if hasattr(e, 'response'):
            print(f"Response: {e.response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
