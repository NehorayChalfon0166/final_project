from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import routers
from routes import health, wallet

# Import model class so it's available for torch.load
from src.models.optimal_gnn import OptimalBitcoinGNN

# Initialize FastAPI app
app = FastAPI(
    title="Bitcoin Wallet Analysis API",
    description="API for analyzing Bitcoin wallet transactions and risk assessment",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    print("✓ Server starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    print("✓ Server shutting down...")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(wallet.router, prefix="/api/v1", tags=["wallet"])

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI Backend"}

if __name__ == "__main__":
    import uvicorn
    print("Starting server on http://localhost:8000")
    print("Docs available at http://localhost:8000/docs")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
