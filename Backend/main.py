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


def get_ssl_config():
    use_https = os.environ.get("USE_HTTPS", "false").lower() == "true"
    keyfile = os.environ.get("SSL_KEYFILE", os.path.join(os.path.dirname(__file__), "privkey.pem"))
    certfile = os.environ.get("SSL_CERTFILE", os.path.join(os.path.dirname(__file__), "fullchain.pem"))

    if use_https and os.path.exists(keyfile) and os.path.exists(certfile):
        return {
            "ssl_keyfile": keyfile,
            "ssl_certfile": certfile,
        }

    return {}

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
        "https://CryptoTrace.cs.bgu.ac.il",
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
    ssl_config = get_ssl_config()
    default_port = "443" if ssl_config else "8000"
    port = int(os.environ.get("API_PORT", default_port))
    scheme = "https" if ssl_config else "http"
    print(f"Starting server on {scheme}://{host}:{port}")
    print(f"Docs available at {scheme}://{host}:{port}/docs")
    uvicorn.run("main:app", host=host, port=port, reload=debug, **ssl_config)
