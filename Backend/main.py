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
    try:
        from routes.utils.model_fit_utils import get_cached_model
        get_cached_model()
        print("✓ Model preloaded into cache")
    except Exception as e:
        print(f"[!] Model preload failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    print("✓ Server shutting down...")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
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
    debug = os.environ.get("DEBUG", "true").lower() == "true"
    host = os.environ.get("API_HOST", "127.0.0.1")
    port = int(os.environ.get("API_PORT", "8000"))
    print(f"Starting server on http://{host}:{port}")
    print(f"Docs available at http://{host}:{port}/docs")
    uvicorn.run("main:app", host=host, port=port, reload=debug)
