from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Import routers
from routes import health, wallet, transactions

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup event
    print("Application starting up...")
    yield
    # Shutdown event
    print("Application shutting down...")

# Initialize FastAPI app
app = FastAPI(
    title="My API",
    description="A FastAPI backend template",
    version="1.0.0",
    lifespan=lifespan
)

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
app.include_router(transactions.router, prefix="/api/v1", tags=["transactions"])

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI Backend"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
