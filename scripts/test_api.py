#!/usr/bin/env python3
"""
Test script for the RAG API
"""

import requests
import json
import time

API_BASE = "http://127.0.0.1:8000"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    response = requests.get(f"{API_BASE}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_initialize():
    """Test initialization endpoint"""
    print("🚀 Testing initialization endpoint...")
    response = requests.post(f"{API_BASE}/initialize")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_query():
    """Test query endpoint"""
    print("❓ Testing query endpoint...")

    payload = {
        "question": "What is RAG?",
        "strategy": "hybrid"
    }

    response = requests.post(f"{API_BASE}/query", json=payload)
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Strategy: {result['strategy']}")
        print(f"Processing time: {result['processing_time']}s")
        print(f"Metrics: {result['metrics']}")
    else:
        print(f"Error: {response.text}")
    print()

def test_strategies():
    """Test strategies endpoint"""
    print("🎯 Testing strategies endpoint...")
    response = requests.get(f"{API_BASE}/strategies")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_evaluate():
    """Test evaluate endpoint"""
    print("📊 Testing evaluate endpoint...")

    payload = {
        "question": "What is RAG?",
        "answer": "RAG combines retrieval and generation",
        "contexts": ["RAG is a technique that combines..."]
    }

    response = requests.post(f"{API_BASE}/evaluate", json=payload)
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Metrics: {result['metrics']}")
        print(f"Processing time: {result['processing_time']}s")
    else:
        print(f"Error: {response.text}")
    print()

def main():
    """Run all tests"""
    print("🧪 Testing RAG API\n")

    try:
        # Test health (should work even without initialization)
        test_health()

        # Initialize the pipeline
        test_initialize()

        # Test health again (should show pipeline loaded)
        test_health()

        # Test query
        test_query()

        # Test strategies
        test_strategies()

        # Test evaluation
        test_evaluate()

        print("✅ All tests completed!")

    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server. Make sure it's running:")
        print("   python -m uvicorn app.api:app --host 127.0.0.1 --port 8000")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main()