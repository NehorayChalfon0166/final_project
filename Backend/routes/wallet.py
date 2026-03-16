from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import requests
import random
import os
from routes.utils.model_fit_utils import analyze_wallet_pipeline, fetch_transactions_mempool, compute_feature_importance

router = APIRouter()

# Pydantic models
class WalletAnalysis(BaseModel):
    address: str


class AnalysisResult(BaseModel):
    wallet_address: str
    status: str
    nodes_count: Optional[int] = None
    edges_count: Optional[int] = None
    ghost_nodes: Optional[int] = None
    graph_data: Optional[Dict[str, Any]] = None
    classification: Optional[str] = None  # "criminal" or "benign"
    prediction: Optional[List[float]] = None
    risk_score: Optional[float] = None
    confidence: Optional[float] = None
    message: Optional[str] = None
    inference_error: Optional[str] = None


class WalletInfoResult(BaseModel):
    address: str
    transaction_count: int
    transactions: List[Dict[str, Any]]
    balance_sats: Optional[int] = None
    balance_btc: Optional[float] = None
    total_received_sats: Optional[int] = None
    total_sent_sats: Optional[int] = None
    funded_txo_count: Optional[int] = None
    spent_txo_count: Optional[int] = None


class FeatureImportanceResult(BaseModel):
    status: str
    feature_importance: Optional[Dict[str, float]] = None
    num_graphs_used: Optional[int] = None
    message: Optional[str] = None


def fetch_address_stats(address: str) -> Dict[str, Any]:
    """Fetch address statistics from mempool.space API"""
    try:
        response = requests.get(
            f"https://mempool.space/api/address/{address}",
            headers={'User-Agent': 'Mozilla/5.0'},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return {}
    except (requests.RequestException, ValueError, KeyError) as e:
        print(f"Error fetching address stats: {e}")
        return {}


def is_valid_bitcoin_address(address: str) -> bool:
    """Basic format validation for Bitcoin addresses."""
    import re
    # Base58 (P2PKH/P2SH): starts with 1 or 3, 25-34 chars
    # Bech32 (SegWit): starts with bc1, 42-62 chars
    patterns = [
        r'^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$',
        r'^bc1[a-z0-9]{39,59}$',
    ]
    return any(re.match(p, address) for p in patterns)


@router.get("/random")
async def get_random_wallet():
    """
    Get a random wallet address from the latest Bitcoin block.
    
    Returns:
        Random wallet address from recent transactions
    """
    try:
        # Fetch latest block height
        tip_response = requests.get(
            "https://mempool.space/api/blocks/tip/height",
            headers={'User-Agent': 'Mozilla/5.0'},
            timeout=10
        )
        if tip_response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to fetch block height")
        
        tip_height = tip_response.json()
        
        # Get block hash
        block_response = requests.get(
            f"https://mempool.space/api/block-height/{tip_height}",
            headers={'User-Agent': 'Mozilla/5.0'},
            timeout=10
        )
        if block_response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to fetch block hash")
        
        block_hash = block_response.text
        
        # Get transactions from this block
        txs_response = requests.get(
            f"https://mempool.space/api/block/{block_hash}/txs",
            headers={'User-Agent': 'Mozilla/5.0'},
            timeout=10
        )
        if txs_response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to fetch transactions")
        
        txs = txs_response.json()
        
        # Collect all unique addresses from outputs
        addresses = set()
        for tx in txs:
            if 'vout' in tx:
                for output in tx['vout']:
                    addr = output.get('scriptpubkey_address')
                    if addr and not addr.startswith('OP_RETURN'):
                        addresses.add(addr)
        
        # Convert to list and pick random
        if not addresses:
            raise HTTPException(status_code=500, detail="No addresses found in recent block")
        
        address_list = list(addresses)
        random_address = random.choice(address_list)
        
        return {"address": random_address}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching random wallet: {str(e)}")


@router.get("/analyze/{address}", response_model=AnalysisResult)
async def analyze_wallet(address: str):
    """
    Analyze a wallet address: fetch transactions, preprocess, and run inference.

    Args:
        address: Wallet address to analyze

    Returns:
        Analysis results with graph statistics and predictions (JSON)
    """
    if not is_valid_bitcoin_address(address):
        raise HTTPException(status_code=400, detail="Invalid Bitcoin address format")

    # Use server-side model path resolution
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    model_path = os.path.join(project_root, 'outputs', 'gnn_model.pt')

    try:
        result = analyze_wallet_pipeline(address, model_path)
        return AnalysisResult(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Analysis failed: {str(e)}")
    

@router.get("/info/{address}")
async def get_wallet_info(address: str):
    """
    Fetch transaction info for a wallet address.

    Args:
        address: Wallet address to fetch transactions for

    Returns:
        Wallet info with transaction list (JSON)
    """
    if not is_valid_bitcoin_address(address):
        raise HTTPException(status_code=400, detail="Invalid Bitcoin address format")
    try:
        # Fetch transactions and address stats
        transactions = fetch_transactions_mempool(address)
        stats = fetch_address_stats(address)
        
        # Calculate balance from chain_stats and mempool_stats
        chain_stats = stats.get('chain_stats', {})
        mempool_stats = stats.get('mempool_stats', {})
        
        # Funded (received) and spent amounts
        funded_sats = chain_stats.get('funded_txo_sum', 0) + mempool_stats.get('funded_txo_sum', 0)
        spent_sats = chain_stats.get('spent_txo_sum', 0) + mempool_stats.get('spent_txo_sum', 0)
        balance_sats = funded_sats - spent_sats
        
        funded_txo_count = chain_stats.get('funded_txo_count', 0) + mempool_stats.get('funded_txo_count', 0)
        spent_txo_count = chain_stats.get('spent_txo_count', 0) + mempool_stats.get('spent_txo_count', 0)
        
        # Get actual transaction count from the stats (more accurate than len(transactions))
        tx_count = chain_stats.get('tx_count', 0) + mempool_stats.get('tx_count', 0)
        
        return WalletInfoResult(
            address=address,
            transaction_count=tx_count if tx_count > 0 else (len(transactions) if transactions else 0),
            transactions=transactions if transactions else [],
            balance_sats=balance_sats,
            balance_btc=balance_sats / 100_000_000,
            total_received_sats=funded_sats,
            total_sent_sats=spent_sats,
            funded_txo_count=funded_txo_count,
            spent_txo_count=spent_txo_count
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch wallet info: {str(e)}")


@router.get("/feature-importance/{address}", response_model=FeatureImportanceResult)
async def get_feature_importance(address: str):
    """
    Compute feature importance for a specific wallet address.

    Args:
        address: Wallet address to compute feature importance for

    Returns:
        Feature importance scores for each feature
    """
    if not is_valid_bitcoin_address(address):
        raise HTTPException(status_code=400, detail="Invalid Bitcoin address format")
    result = compute_feature_importance(address)
    
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    
    return FeatureImportanceResult(**result)
