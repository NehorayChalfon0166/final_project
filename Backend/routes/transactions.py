from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from enum import Enum

router = APIRouter()

# Enums and Pydantic models
class TransactionStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"

class Transaction(BaseModel):
    id: int
    tx_hash: str
    from_address: str
    to_address: str
    amount: float
    status: TransactionStatus
    timestamp: Optional[str] = None
    is_valid: bool

class TransactionCreate(BaseModel):
    tx_hash: str
    from_address: str
    to_address: str
    amount: float
    status: TransactionStatus = TransactionStatus.PENDING

# In-memory storage for demo
transactions_db: List[Transaction] = []

def validate_transaction(tx_hash: str, from_addr: str, to_addr: str, amount: float) -> bool:
    """
    Basic transaction validation
    Checks for valid format and reasonable values
    """
    # Check if tx_hash is valid hex
    try:
        int(tx_hash, 16)
    except ValueError:
        return False
    
    # Check if addresses are not empty
    if not from_addr or not to_addr:
        return False
    
    # Check if amount is positive
    if amount <= 0:
        return False
    
    return True

@router.get("/transactions", response_model=List[Transaction])
async def list_transactions():
    """Get all transactions"""
    return transactions_db

@router.get("/transactions/{transaction_id}", response_model=Transaction)
async def get_transaction(transaction_id: int):
    """Get a specific transaction by ID"""
    for tx in transactions_db:
        if tx.id == transaction_id:
            return tx
    raise HTTPException(status_code=404, detail="Transaction not found")

@router.post("/transactions/validate")
async def validate_transaction_endpoint(tx: TransactionCreate):
    """Validate a transaction"""
    is_valid = validate_transaction(tx.tx_hash, tx.from_address, tx.to_address, tx.amount)
    return {
        "tx_hash": tx.tx_hash,
        "is_valid": is_valid,
        "message": "Valid transaction" if is_valid else "Invalid transaction format"
    }

@router.post("/transactions", response_model=Transaction)
async def create_transaction(tx: TransactionCreate):
    """Create a new transaction (must be valid)"""
    if not validate_transaction(tx.tx_hash, tx.from_address, tx.to_address, tx.amount):
        raise HTTPException(status_code=400, detail="Invalid transaction format or values")
    
    new_transaction = Transaction(
        id=len(transactions_db) + 1,
        tx_hash=tx.tx_hash,
        from_address=tx.from_address,
        to_address=tx.to_address,
        amount=tx.amount,
        status=tx.status,
        timestamp="2024-12-24",
        is_valid=True
    )
    transactions_db.append(new_transaction)
    return new_transaction

@router.put("/transactions/{transaction_id}", response_model=Transaction)
async def update_transaction(transaction_id: int, tx: TransactionCreate):
    """Update an existing transaction"""
    if not validate_transaction(tx.tx_hash, tx.from_address, tx.to_address, tx.amount):
        raise HTTPException(status_code=400, detail="Invalid transaction format or values")
    
    for idx, existing_tx in enumerate(transactions_db):
        if existing_tx.id == transaction_id:
            updated_tx = Transaction(
                id=transaction_id,
                tx_hash=tx.tx_hash,
                from_address=tx.from_address,
                to_address=tx.to_address,
                amount=tx.amount,
                status=tx.status,
                timestamp=existing_tx.timestamp,
                is_valid=True
            )
            transactions_db[idx] = updated_tx
            return updated_tx
    raise HTTPException(status_code=404, detail="Transaction not found")

@router.delete("/transactions/{transaction_id}")
async def delete_transaction(transaction_id: int):
    """Delete a transaction"""
    for idx, tx in enumerate(transactions_db):
        if tx.id == transaction_id:
            transactions_db.pop(idx)
            return {"message": "Transaction deleted"}
    raise HTTPException(status_code=404, detail="Transaction not found")
