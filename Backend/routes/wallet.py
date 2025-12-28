from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import re
from datetime import datetime
from ..utils.model_fit_utils import analyze_wallet_pipeline

router = APIRouter()

# Pydantic models for CRUD operations
class Wallet(BaseModel):
    id: int
    address: str
    is_valid: bool
    balance: Optional[float] = None
    risk_score: Optional[float] = None
    last_analyzed: Optional[str] = None
    created_at: Optional[str] = None

class WalletCreate(BaseModel):
    address: str
    balance: Optional[float] = None

# Pydantic models for analysis
class GraphData(BaseModel):
    x_shape: tuple
    y_shape: tuple
    edge_index_shape: tuple
    edge_attr_shape: tuple

class AnalysisResult(BaseModel):
    wallet_address: str
    status: str
    nodes_count: int
    edges_count: int
    graph_data: Dict[str, Any]
    prediction: Optional[List[float]] = None
    risk_score: Optional[float] = None
    message: Optional[str] = None
    inference_error: Optional[str] = None

# In-memory storage for demo (replace with actual database in production)
wallets_db: List[Wallet] = []

def validate_wallet_address(address: str) -> bool:
    """
    Validate wallet address format
    Supports Ethereum, Bitcoin, and generic hex addresses
    """
    if re.match(r'^0x[a-fA-F0-9]{40}$', address):  # Ethereum
        return True
    if re.match(r'^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$', address):  # Bitcoin
        return True
    if re.match(r'^[a-fA-F0-9]{40,64}$', address):  # Generic hex
        return True
    return False

# ============================================
# CRUD ENDPOINTS (Wallet Management)
# ============================================

@router.get("/wallets", response_model=List[Wallet])
async def list_wallets():
    """Get all stored wallets"""
    return wallets_db

@router.get("/wallets/{wallet_id}", response_model=Wallet)
async def get_wallet(wallet_id: int):
    """Get a specific wallet by ID"""
    for wallet in wallets_db:
        if wallet.id == wallet_id:
            return wallet
    raise HTTPException(status_code=404, detail="Wallet not found")

@router.post("/wallets/validate")
async def validate_wallet(wallet: WalletCreate):
    """Validate a wallet address format"""
    is_valid = validate_wallet_address(wallet.address)
    return {
        "address": wallet.address,
        "is_valid": is_valid,
        "message": "Valid wallet address" if is_valid else "Invalid wallet address format"
    }

@router.post("/wallets", response_model=Wallet)
async def create_wallet(wallet: WalletCreate):
    """Create a new wallet record (address must be valid)"""
    if not validate_wallet_address(wallet.address):
        raise HTTPException(status_code=400, detail="Invalid wallet address format")

    # Check if wallet already exists
    for existing_wallet in wallets_db:
        if existing_wallet.address == wallet.address:
            raise HTTPException(status_code=400, detail="Wallet address already exists")

    new_wallet = Wallet(
        id=len(wallets_db) + 1,
        address=wallet.address,
        is_valid=True,
        balance=wallet.balance or 0.0,
        created_at=datetime.now().isoformat()
    )
    wallets_db.append(new_wallet)
    return new_wallet

@router.put("/wallets/{wallet_id}", response_model=Wallet)
async def update_wallet(wallet_id: int, wallet: WalletCreate):
    """Update an existing wallet"""
    if not validate_wallet_address(wallet.address):
        raise HTTPException(status_code=400, detail="Invalid wallet address format")

    for idx, existing_wallet in enumerate(wallets_db):
        if existing_wallet.id == wallet_id:
            updated_wallet = Wallet(
                id=wallet_id,
                address=wallet.address,
                is_valid=True,
                balance=wallet.balance or existing_wallet.balance,
                risk_score=existing_wallet.risk_score,
                last_analyzed=existing_wallet.last_analyzed,
                created_at=existing_wallet.created_at
            )
            wallets_db[idx] = updated_wallet
            return updated_wallet
    raise HTTPException(status_code=404, detail="Wallet not found")

@router.delete("/wallets/{wallet_id}")
async def delete_wallet(wallet_id: int):
    """Delete a wallet"""
    for idx, wallet in enumerate(wallets_db):
        if wallet.id == wallet_id:
            wallets_db.pop(idx)
            return {"message": "Wallet deleted successfully"}
    raise HTTPException(status_code=404, detail="Wallet not found")

# ============================================
# ANALYSIS ENDPOINT (Real-time Risk Analysis)
# ============================================

@router.post("/analyze/{address}", response_model=AnalysisResult)
async def analyze_wallet(address: str, model_path: Optional[str] = None, save_to_db: bool = True):
    """
    Analyze a wallet address: fetch transactions, preprocess, and run inference.
    
    Args:
        address: Wallet address to analyze
        model_path: Optional path to saved model for inference
        save_to_db: Whether to save/update the wallet in database after analysis
    
    Returns:
        Analysis results with graph statistics and risk predictions (JSON)
    """
    if not validate_wallet_address(address):
        raise HTTPException(status_code=400, detail="Invalid wallet address format")
    
    try:
        # Run the analysis pipeline
        result = analyze_wallet_pipeline(address, model_path)
        
        # Optionally save/update wallet in database
        if save_to_db and result.get('status') == 'success':
            risk_score = result.get('risk_score')
            
            # Check if wallet already exists
            wallet_found = False
            for idx, wallet in enumerate(wallets_db):
                if wallet.address == address:
                    # Update existing wallet
                    wallets_db[idx].risk_score = risk_score
                    wallets_db[idx].last_analyzed = datetime.now().isoformat()
                    wallet_found = True
                    break
            
            # Create new wallet if not found
            if not wallet_found:
                new_wallet = Wallet(
                    id=len(wallets_db) + 1,
                    address=address,
                    is_valid=True,
                    risk_score=risk_score,
                    last_analyzed=datetime.now().isoformat(),
                    created_at=datetime.now().isoformat()
                )
                wallets_db.append(new_wallet)
        
        return AnalysisResult(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Analysis failed: {str(e)}")
