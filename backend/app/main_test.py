"""
Minimal test server to diagnose blocking issues.
Run with: python -m uvicorn app.main_test:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time

app = FastAPI(title="FinAgent Test")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
async def ping():
    return {"status": "pong", "time": time.time()}

@app.get("/")
async def root():
    return {"message": "Test server working"}

print("Minimal test server loaded successfully")
