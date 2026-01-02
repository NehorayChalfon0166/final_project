from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import requests
from routes.utils.model_fit_utils import analyze_wallet_pipeline

router = APIRouter()

# Pydantic models
class WalletAnalysis(BaseModel):
    address: str


class AnalysisResult(BaseModel):
    wallet_address: str
    status: str
    nodes_count: Optional[int] = None
    edges_count: Optional[int] = None
    graph_data: Optional[Dict[str, Any]] = None
    classification: Optional[str] = None  # "criminal" or "benign"
    prediction: Optional[List[float]] = None
    risk_score: Optional[float] = None
    confidence: Optional[float] = None
    message: Optional[str] = None
    inference_error: Optional[str] = None


async def validate_wallet_address(address: str) -> bool:
    """Validate Bitcoin wallet address using mempool.space API"""
    try:
        response = requests.get(
            f"https://mempool.space/api/v1/validate-address/{address}",
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            return data.get('isvalid', False)
        return False
    except:
        return False

@router.get("/analyze/{address}", response_model=AnalysisResult)
async def analyze_wallet(address: str, model_path: Optional[str] = "../models/crypto_gnn_model.pt"):
    """
    Analyze a wallet address: fetch transactions, preprocess, and run inference.
    
    Args:
        address: Wallet address to analyze
        model_path: Optional path to saved model (defaults to crypto_gnn_model.pt)
    
    Returns:
        Analysis results with graph statistics and predictions (JSON)
    """
    # Validate address using mempool.space API
    is_valid = await validate_wallet_address(address)
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"Invalid or inactive Bitcoin address: {address}")
    
    try:
        result = analyze_wallet_pipeline(address, model_path)
        return AnalysisResult(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Analysis failed: {str(e)}")
