#!/usr/bin/env python3
"""
Debug dotenv loading
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent.parent / "backend" / ".env"
print(f"Loading .env from: {env_path}")
print(f".env exists: {env_path.exists()}")

result = load_dotenv(env_path)
print(f"load_dotenv result: {result}")

# Check environment variables
import os
print(f"\nEnvironment variables:")
print(f"OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY', 'NOT SET')[:20] if os.getenv('OPENAI_API_KEY') else 'NOT SET'}...")
print(f"COHERE_API_KEY: {os.getenv('COHERE_API_KEY', 'NOT SET')[:20] if os.getenv('COHERE_API_KEY') else 'NOT SET'}...")
