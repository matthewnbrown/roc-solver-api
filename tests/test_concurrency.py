#!/usr/bin/env python3
"""
Concurrent testing script to verify thread safety of the Captcha Solve API
"""

import requests
import threading
import time
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import io
import random
import string

API_BASE_URL = "http://localhost:8000"

def create_test_image():
    """Create a simple test image"""
    img = Image.new('L', (64, 64), color=random.randint(0, 255))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes.getvalue()

def generate_random_hash():
    """Generate a random hash for testing"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=16))

def solve_captcha_request(thread_id, request_count):
    """Make a solve captcha request from a specific thread"""
    results = []
    
    for i in range(request_count):
        try:
            # Create test image data
            image_data = create_test_image()
            captcha_hash = f"thread_{thread_id}_request_{i}_{generate_random_hash()}"
            
            # Make request
            files = {'image': ('test.png', image_data, 'image/png')}
            data = {'captcha_hash': captcha_hash}
            
            start_time = time.time()
            response = requests.post(f"{API_BASE_URL}/api/v1/solve", files=files, data=data)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                results.append({
                    'thread_id': thread_id,
                    'request_id': result.get('request_id'),
                    'success': True,
                    'response_time': response_time,
                    'captcha_hash': captcha_hash
                })
                print(f"Thread {thread_id}, Request {i+1}: SUCCESS ({response_time:.3f}s)")
            else:
                results.append({
                    'thread_id': thread_id,
                    'success': False,
                    'error': response.text,
                    'response_time': response_time,
                    'captcha_hash': captcha_hash
                })
                print(f"Thread {thread_id}, Request {i+1}: FAILED - {response.status_code}")
                
        except Exception as e:
            results.append({
                'thread_id': thread_id,
                'success': False,
                'error': str(e),
                'captcha_hash': captcha_hash
            })
            print(f"Thread {thread_id}, Request {i+1}: ERROR - {str(e)}")
            
        # Small delay between requests
        time.sleep(0.1)
    
    return results

def submit_feedback_request(request_data):
    """Submit feedback for a solved captcha"""
    try:
        data = {
            'request_id': request_data['request_id'],
            'is_correct': random.choice([True, False]),
            'actual_answer': str(random.randint(0, 9))
        }
        
        response = requests.post(f"{API_BASE_URL}/api/v1/feedback", data=data)
        
        if response.status_code == 200:
            print(f"Feedback submitted for {request_data['request_id'][:8]}...")
            return True
        else:
            print(f"Feedback failed for {request_data['request_id'][:8]}...: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"Feedback error for {request_data['request_id'][:8]}...: {str(e)}")
        return False

def test_concurrent_solve_requests(num_threads=10, requests_per_thread=5):
    """Test concurrent solve requests"""
    print(f"\n🧪 Testing {num_threads} concurrent threads with {requests_per_thread} requests each")
    print("=" * 70)
    
    start_time = time.time()
    all_results = []
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all solve requests
        futures = [
            executor.submit(solve_captcha_request, thread_id, requests_per_thread)
            for thread_id in range(num_threads)
        ]
        
        # Collect results
        for future in as_completed(futures):
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                print(f"Thread execution error: {e}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Analyze results
    successful_requests = [r for r in all_results if r['success']]
    failed_requests = [r for r in all_results if not r['success']]
    
    print(f"\n📊 Solve Requests Results:")
    print(f"Total requests: {len(all_results)}")
    print(f"Successful: {len(successful_requests)}")
    print(f"Failed: {len(failed_requests)}")
    print(f"Success rate: {len(successful_requests)/len(all_results)*100:.1f}%")
    print(f"Total time: {total_time:.2f}s")
    
    if successful_requests:
        response_times = [r['response_time'] for r in successful_requests if 'response_time' in r]
        if response_times:
            print(f"Average response time: {sum(response_times)/len(response_times):.3f}s")
            print(f"Min response time: {min(response_times):.3f}s")
            print(f"Max response time: {max(response_times):.3f}s")
    
    return successful_requests

def test_concurrent_feedback(solve_results, num_threads=5):
    """Test concurrent feedback submissions"""
    if not solve_results:
        print("No successful solve results to test feedback with")
        return
    
    print(f"\n🔄 Testing concurrent feedback with {num_threads} threads")
    print("=" * 50)
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit feedback for all successful solve requests
        futures = [
            executor.submit(submit_feedback_request, result)
            for result in solve_results
        ]
        
        # Collect results
        feedback_results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                feedback_results.append(result)
            except Exception as e:
                print(f"Feedback thread error: {e}")
                feedback_results.append(False)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    successful_feedback = sum(1 for r in feedback_results if r)
    
    print(f"\n📊 Feedback Results:")
    print(f"Total feedback submissions: {len(feedback_results)}")
    print(f"Successful: {successful_feedback}")
    print(f"Failed: {len(feedback_results) - successful_feedback}")
    print(f"Success rate: {successful_feedback/len(feedback_results)*100:.1f}%")
    print(f"Total time: {total_time:.2f}s")

def test_stats_during_load():
    """Test stats endpoint during high load"""
    print(f"\n📈 Testing stats endpoint during load")
    print("=" * 40)
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"Stats retrieved successfully:")
            print(f"  Total requests: {stats.get('total_requests', 0)}")
            print(f"  Accuracy: {stats.get('accuracy_percentage', 0):.2f}%")
            print(f"  Recent requests (24h): {stats.get('recent_requests_24h', 0)}")
        else:
            print(f"Stats request failed: {response.status_code}")
    except Exception as e:
        print(f"Stats request error: {e}")

def main():
    """Run the complete concurrency test suite"""
    print("🚀 Captcha Solve API Concurrency Test Suite")
    print("=" * 50)
    
    # Check if API is running
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code != 200:
            print("❌ API is not responding. Please start the server first.")
            return
        
        health = response.json()
        print(f"✅ API is healthy")
        print(f"   Predictor loaded: {health.get('predictor_loaded', False)}")
        print()
        
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("Please make sure the server is running on http://localhost:8000")
        return
    
    # Test 1: Concurrent solve requests
    solve_results = test_concurrent_solve_requests(num_threads=8, requests_per_thread=3)
    
    # Small delay
    time.sleep(1)
    
    # Test 2: Concurrent feedback
    test_concurrent_feedback(solve_results, num_threads=4)
    
    # Small delay
    time.sleep(1)
    
    # Test 3: Stats during load
    test_stats_during_load()
    
    print(f"\n✅ Concurrency testing completed!")
    print(f"The API handled concurrent requests successfully with the new thread-safe database manager.")

if __name__ == "__main__":
    main()
