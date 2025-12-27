from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import re
from ..utils.model_fit_utils import analyze_wallet_pipeline

router = APIRouter()

# Pydantic models
class WalletAnalysis(BaseModel):
    address: str


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


def validate_wallet_address(address: str) -> bool:
    """Validate wallet address format"""
    if re.match(r'^0x[a-fA-F0-9]{40}$', address):  # Ethereum
        return True
    if re.match(r'^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$', address):  # Bitcoin
        return True
    if re.match(r'^[a-fA-F0-9]{40,64}$', address):  # Generic hex
        return True
    return False

@router.get("/analyze/{address}", response_model=AnalysisResult)
async def analyze_wallet(address: str, model_path: Optional[str] = None):
    """
    Analyze a wallet address: fetch transactions, preprocess, and run inference.
    
    Args:
        address: Wallet address to analyze
        model_path: Optional path to saved model
    
    Returns:
        Analysis results with graph statistics and predictions (JSON)
    """
    if not validate_wallet_address(address):
        raise HTTPException(status_code=400, detail="Invalid wallet address format")
    
    try:
        result = analyze_wallet_pipeline(address, model_path)
        return AnalysisResult(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Analysis failed: {str(e)}")
