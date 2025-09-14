#!/usr/bin/env python3
"""
Test script for the Captcha Solve API
"""

import requests
import json
import os
from PIL import Image
import io

API_BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_solve_captcha(image_path, captcha_hash="test123"):
    """Test the solve captcha endpoint"""
    print(f"\nTesting solve captcha endpoint with {image_path}...")
    
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return None
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {'captcha_hash': captcha_hash}
            
            response = requests.post(f"{API_BASE_URL}/solve", files=files, data=data)
            
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            return result
        else:
            print(f"Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_feedback(request_id, is_correct=True, actual_answer="5"):
    """Test the feedback endpoint"""
    print(f"\nTesting feedback endpoint...")
    
    try:
        data = {
            'request_id': request_id,
            'is_correct': is_correct,
            'actual_answer': actual_answer
        }
        
        response = requests.post(f"{API_BASE_URL}/feedback", data=data)
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            return result
        else:
            print(f"Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_stats():
    """Test the stats endpoint"""
    print(f"\nTesting stats endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            return result
        else:
            print(f"Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        return None

def create_test_image():
    """Create a simple test image for testing"""
    # Create a simple 64x64 grayscale image with a digit
    img = Image.new('L', (64, 64), color=255)  # White background
    
    # You could add some simple drawing here to simulate a captcha
    # For now, just return a blank image
    test_path = "test_captcha.png"
    img.save(test_path)
    return test_path

def main():
    """Run all tests"""
    print("Captcha Solve API Test Suite")
    print("=" * 40)
    
    # Test health endpoint
    if not test_health():
        print("Health check failed. Make sure the server is running.")
        return
    
    # Create a test image
    test_image = create_test_image()
    print(f"Created test image: {test_image}")
    
    # Test solve endpoint
    solve_result = test_solve_captcha(test_image)
    
    if solve_result:
        request_id = solve_result.get('request_id')
        
        # Test feedback endpoint
        test_feedback(request_id, is_correct=True, actual_answer="5")
        
        # Test stats endpoint
        test_stats()
    
    # Clean up test image
    if os.path.exists(test_image):
        os.remove(test_image)
        print(f"\nCleaned up test image: {test_image}")
    
    print("\nTest suite completed!")

if __name__ == "__main__":
    main()
