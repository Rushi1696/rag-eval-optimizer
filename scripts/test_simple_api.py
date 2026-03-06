#!/usr/bin/env python3
"""
Test script for simple RAG API
"""

import requests
import json

API_BASE = "http://127.0.0.1:8000"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing health...")
    response = requests.get(f"{API_BASE}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_query():
    """Test query endpoint"""
    print("❓ Testing query...")

    payload = {"question": "What is RAG?"}

    response = requests.post(f"{API_BASE}/query", json=payload)
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Time: {result['processing_time']}s")
    else:
        print(f"Error: {response.text}")
    print()

def main():
    """Run tests"""
    print("🧪 Testing Simple RAG API\n")

    try:
        test_health()
        test_query()
        print("✅ Tests completed!")

    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server. Start with:")
        print("   python -m uvicorn app.simple_api:app --host 127.0.0.1 --port 8000")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main()