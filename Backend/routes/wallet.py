from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import re

router = APIRouter()

# Pydantic models
class Wallet(BaseModel):
    id: int
    address: str
    is_valid: bool
    balance: Optional[float] = None
    created_at: Optional[str] = None

class WalletCreate(BaseModel):
    address: str
    balance: Optional[float] = None

# In-memory storage for demo
wallets_db: List[Wallet] = []

def validate_wallet_address(address: str) -> bool:
    """
    Basic wallet address validation
    Checks for common address formats (Ethereum, Bitcoin, etc.)
    """
    # Ethereum address format (0x followed by 40 hex characters)
    if re.match(r'^0x[a-fA-F0-9]{40}$', address):
        return True
    # Bitcoin address format
    if re.match(r'^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$', address):
        return True
    # Generic hex address
    if re.match(r'^[a-fA-F0-9]{40,64}$', address):
        return True
    return False

@router.get("/wallets", response_model=List[Wallet])
async def list_wallets():
    """Get all wallets"""
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
    """Validate a wallet address"""
    is_valid = validate_wallet_address(wallet.address)
    return {
        "address": wallet.address,
        "is_valid": is_valid,
        "message": "Valid wallet address" if is_valid else "Invalid wallet address format"
    }

@router.post("/wallets", response_model=Wallet)
async def create_wallet(wallet: WalletCreate):
    """Create a new wallet (address must be valid)"""
    if not validate_wallet_address(wallet.address):
        raise HTTPException(status_code=400, detail="Invalid wallet address format")
    
    new_wallet = Wallet(
        id=len(wallets_db) + 1,
        address=wallet.address,
        is_valid=True,
        balance=wallet.balance or 0.0,
        created_at="2024-12-24"
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
                balance=wallet.balance or 0.0,
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
            return {"message": "Wallet deleted"}
    raise HTTPException(status_code=404, detail="Wallet not found")
