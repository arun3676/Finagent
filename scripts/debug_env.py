#!/usr/bin/env python3
"""
Debug environment variables loading
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.config import settings

print("Debug: Environment Variables")
print("="*60)
print(f"OPENAI_API_KEY: {settings.OPENAI_API_KEY[:10] if settings.OPENAI_API_KEY else 'NOT SET'}...")
print(f"COHERE_API_KEY: {settings.COHERE_API_KEY[:10] if settings.COHERE_API_KEY else 'NOT SET'}...")
print(f"QDRANT_HOST: {settings.QDRANT_HOST}")
print(f"LLM_MODEL: {settings.LLM_MODEL}")
print(f"EMBEDDING_MODEL: {settings.EMBEDDING_MODEL}")

# Check if .env file exists and has content
env_file = Path(__file__).parent.parent / "backend" / ".env"
print(f"\n.env file exists: {env_file.exists()}")
if env_file.exists():
    with open(env_file, 'r') as f:
        lines = f.readlines()[:5]  # First 5 lines
        print(f"\nFirst 5 lines of .env:")
        for line in lines:
            if '=' in line:
                key, value = line.split('=', 1)
                if 'KEY' in key:
                    print(f"  {key}=***masked***")
                else:
                    print(f"  {key}={value.strip()}")
