from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

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

FRONTEND_DIST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Frontend", "dist"))
FRONTEND_INDEX_FILE = os.path.join(FRONTEND_DIST_DIR, "index.html")


def get_ssl_config():
    requested_port = os.environ.get("API_PORT")
    use_https = os.environ.get("USE_HTTPS", "false").lower() == "true" or requested_port == "443"
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
        "https://www.CryptoTrace.cs.bgu.ac.il",
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

if os.path.isfile(FRONTEND_INDEX_FILE):

    @app.get("/{requested_path:path}", include_in_schema=False)
    async def serve_frontend(requested_path: str):
        if requested_path.startswith("api/") or requested_path in {"docs", "redoc", "openapi.json"}:
            raise HTTPException(status_code=404)

        if requested_path:
            candidate_path = os.path.abspath(os.path.join(FRONTEND_DIST_DIR, requested_path))
            if candidate_path.startswith(FRONTEND_DIST_DIR + os.sep) and os.path.isfile(candidate_path):
                return FileResponse(candidate_path)

        return FileResponse(FRONTEND_INDEX_FILE)

else:

    @app.get("/")
    async def root():
        return {"message": "Welcome to the FastAPI Backend"}

if __name__ == "__main__":
    import uvicorn
    debug = os.environ.get("DEBUG", "true").lower() == "true"
    host = os.environ.get("API_HOST", "0.0.0.0")
    ssl_config = get_ssl_config()
    default_port = "443" if ssl_config else "8000"
    port = int(os.environ.get("API_PORT", default_port))
    scheme = "https" if ssl_config else "http"
    print(f"Starting server on {scheme}://{host}:{port}")
    print(f"Docs available at {scheme}://{host}:{port}/docs")
    uvicorn.run("main:app", host=host, port=port, reload=debug, **ssl_config)
