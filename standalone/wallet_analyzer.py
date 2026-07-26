#!/usr/bin/env python3
"""
Bitcoin Wallet Risk Analyzer — Standalone one-click analyzer
============================================================

A single, self-contained script. Run it and it will:

  1. Install every library it needs (numpy, torch, torch-geometric, requests).
  2. Load the trained GATv2 GNN model (found locally inside the repo, or
     downloaded automatically from GitHub if you copied this file elsewhere).
  3. Fetch a wallet's live transaction history from mempool.space.
  4. Print a report with:
        - a RISK PERCENTAGE  (chance the wallet is criminal)
        - money coming IN / going OUT
        - total money currently in the wallet (BTC + USD)
        - the TOP 3 features that drove the model's decision

Usage
-----
    python3 wallet_analyzer.py                 # prompts for an address
    python3 wallet_analyzer.py <BTC_ADDRESS>   # analyze directly
    python3 wallet_analyzer.py <ADDR> --json   # machine-readable output

Or just double-click `run.command` (macOS) / `run.bat` (Windows) — those set
up an isolated environment for you so nothing touches your system Python.

The model architecture and feature engineering below are vendored copies of the
project's canonical implementation (src/models/optimal_gnn.py,
src/graph/graph_builder.py, src/graph/config.py) so this file stays portable.
"""

# =============================================================================
# SECTION 0 — Dependency bootstrap (runs before any heavy import)
# =============================================================================
import importlib.util
import subprocess
import sys

MIN_PYTHON = (3, 9)

# (import name, pip install spec) — order matters: numpy + torch must land
# before torch-geometric, which is built against them.
REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("torch_geometric", "torch-geometric"),
    ("requests", "requests"),
]


def _ensure_dependencies() -> None:
    """Install any missing third-party packages into the current environment."""
    if sys.version_info < MIN_PYTHON:
        sys.exit(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required "
            f"(you have {sys.version_info.major}.{sys.version_info.minor})."
        )

    missing = [
        (mod, spec)
        for mod, spec in REQUIRED_PACKAGES
        if importlib.util.find_spec(mod) is None
    ]
    if not missing:
        return

    print("=" * 60)
    print("  First-time setup — installing required libraries")
    print("  (torch is large; this can take a few minutes)")
    print("=" * 60)
    for mod, spec in missing:
        print(f"  → installing {spec} ...", flush=True)
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet",
                 "--disable-pip-version-check", spec]
            )
        except subprocess.CalledProcessError as exc:
            sys.exit(
                f"\nFailed to install '{spec}'. Try running manually:\n"
                f"    {sys.executable} -m pip install {spec}\n({exc})"
            )
    importlib.invalidate_caches()
    print("  ✓ All libraries installed.\n")


_ensure_dependencies()

# ---- Safe to import the heavy stack now ------------------------------------
import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import (
    GATv2Conv,
    LayerNorm,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.utils import dropout_edge


# =============================================================================
# SECTION 1 — Configuration (mirrors src/graph/config.py)
# =============================================================================
MEMPOOL_API = "https://mempool.space/api"
SAT_TO_BTC = 1e-8

# Where to grab the trained model if it isn't found on disk. Tracks main, which
# carries the latest retrained model (use-everything split, 77,729 wallets;
# GNN ROC-AUC ~0.958).
REPO_RAW_BASE = (
    "https://raw.githubusercontent.com/"
    "NehorayChalfon0166/final_project/main"
)
MODEL_URL = f"{REPO_RAW_BASE}/outputs/gnn_model.pt"
TEMPERATURE_URL = f"{REPO_RAW_BASE}/outputs/temperature.pt"

# 12 selected node features, in the exact order the model was trained on.
FEATURE_COLUMNS = [
    "lifetime_seconds_log",
    "activity_rate_log",
    "in_out_balance_log",
    "total_txs_log",
    "send_receive_ratio_log",
    "fee_per_tx_log",
    "blocks_btwn_txs_mean_log",
    "fee_share_mean_log",
    "avg_tx_size_log",
    "tx_size_range_log",
    "max_sent_log",
    "max_received_log",
]
NUM_NODE_FEATURES = len(FEATURE_COLUMNS)  # 12
NUM_EDGE_FEATURES = 3  # amount_log, direction, timestamp_norm

# Human-readable names for the report.
FEATURE_DISPLAY = {
    "lifetime_seconds_log": "Wallet lifetime (active time span)",
    "activity_rate_log": "Activity rate (transactions per day)",
    "in_out_balance_log": "Incoming-vs-outgoing transaction ratio",
    "total_txs_log": "Total number of transactions",
    "send_receive_ratio_log": "Sent-to-received volume ratio",
    "fee_per_tx_log": "Average fee paid per transaction",
    "blocks_btwn_txs_mean_log": "Average blocks between transactions",
    "fee_share_mean_log": "Fee as a share of amount moved",
    "avg_tx_size_log": "Average transaction size",
    "tx_size_range_log": "Spread between largest & smallest send",
    "max_sent_log": "Largest single amount sent",
    "max_received_log": "Largest single amount received",
}

# Timestamp normalization bounds (Bitcoin genesis → ~2030).
TIMESTAMP_MIN = 1231006505
TIMESTAMP_MAX = 1893456000


# =============================================================================
# SECTION 2 — Model (vendored from src/models/optimal_gnn.py, inference only)
# =============================================================================
class OptimalBitcoinGNN(nn.Module):
    """3-layer GATv2 with ghost-node embeddings, residuals and hybrid readout."""

    def __init__(
        self,
        num_node_features: int = 12,
        num_edge_features: int = 3,
        hidden_dim: int = 64,
        num_heads_1: int = 4,
        num_heads_2: int = 4,
        num_heads_3: int = 2,
        dropout: float = 0.2,
        final_dropout: float = 0.3,
        drop_edge_rate: float = 0.1,
    ):
        super().__init__()
        self.num_node_features = num_node_features
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.drop_edge_rate = drop_edge_rate

        # Ghost node handling: learnable embedding + node type.
        self.ghost_embedding = nn.Parameter(torch.randn(1, num_node_features) * 0.01)
        self.node_type_embedding = nn.Embedding(2, 8)  # 0=center, 1=ghost
        self.input_proj = nn.Linear(num_node_features + 1 + 8, num_node_features)

        self.conv1 = GATv2Conv(
            in_channels=num_node_features, out_channels=hidden_dim,
            heads=num_heads_1, edge_dim=num_edge_features, concat=True, dropout=dropout,
        )
        self.norm1 = LayerNorm(hidden_dim * num_heads_1)

        self.conv2 = GATv2Conv(
            in_channels=hidden_dim * num_heads_1, out_channels=hidden_dim,
            heads=num_heads_2, edge_dim=num_edge_features, concat=True, dropout=dropout,
        )
        self.norm2 = LayerNorm(hidden_dim * num_heads_2)
        self.residual_proj = nn.Linear(hidden_dim * num_heads_1, hidden_dim * num_heads_2)

        self.conv3 = GATv2Conv(
            in_channels=hidden_dim * num_heads_2, out_channels=hidden_dim // 2,
            heads=num_heads_3, edge_dim=num_edge_features, concat=True, dropout=dropout * 0.5,
        )
        self.norm3 = LayerNorm((hidden_dim // 2) * num_heads_3)

        final_gnn_dim = (hidden_dim // 2) * num_heads_3  # 64
        self.initial_proj = nn.Linear(num_node_features, final_gnn_dim)

        classifier_input_dim = final_gnn_dim * 3  # center + mean + max = 192
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(final_dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.dropout_layer = nn.Dropout(dropout)
        self.dropout_light = nn.Dropout(dropout * 0.5)

    def _prepare_node_features(self, x):
        """Replace zero-feature ghost nodes with a learned embedding + type info."""
        has_features = (x.abs().sum(dim=1) > 0).float()
        ghost_mask = has_features == 0
        if ghost_mask.any():
            x = x.clone()
            x[ghost_mask] = self.ghost_embedding.expand(ghost_mask.sum(), -1)
        node_types = ghost_mask.long()
        type_emb = self.node_type_embedding(node_types)
        x_augmented = torch.cat([x, has_features.unsqueeze(1), type_emb], dim=1)
        return self.input_proj(x_augmented)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self._prepare_node_features(x)
        x_init = x

        if self.training and self.drop_edge_rate > 0 and edge_index.size(1) > 0:
            edge_index_drop, edge_mask = dropout_edge(
                edge_index, p=self.drop_edge_rate, training=self.training
            )
            edge_attr_drop = edge_attr[edge_mask] if edge_attr is not None else None
        else:
            edge_index_drop = edge_index
            edge_attr_drop = edge_attr

        h = self.conv1(x, edge_index_drop, edge_attr=edge_attr_drop)
        h = self.dropout_layer(F.elu(self.norm1(h)))
        h_residual = h

        h = self.conv2(h, edge_index_drop, edge_attr=edge_attr_drop)
        h = self.dropout_layer(F.elu(self.norm2(h)))
        h = h + self.residual_proj(h_residual)

        h = self.conv3(h, edge_index, edge_attr=edge_attr)
        h = self.dropout_light(F.elu(self.norm3(h)))
        h = h + self.initial_proj(x_init)

        center_emb = self._get_center_embeddings(h, batch)
        mean_emb = global_mean_pool(h, batch)
        max_emb = global_max_pool(h, batch)
        h_readout = torch.cat([center_emb, mean_emb, max_emb], dim=1)
        return self.classifier(h_readout)

    def _get_center_embeddings(self, h, batch):
        """Center node is index 0 within each graph (by construction)."""
        counts = torch.bincount(batch)
        ptr = torch.zeros(counts.size(0) + 1, dtype=torch.long, device=batch.device)
        torch.cumsum(counts, dim=0, out=ptr[1:])
        return h[ptr[:-1]]


# =============================================================================
# SECTION 3 — Feature engineering + graph building
# (vendored from src/graph/graph_builder.py)
# =============================================================================
def _normalize_timestamp(ts: int) -> float:
    if ts <= TIMESTAMP_MIN:
        return 0.0
    if ts >= TIMESTAMP_MAX:
        return 1.0
    return (ts - TIMESTAMP_MIN) / (TIMESTAMP_MAX - TIMESTAMP_MIN)


def _log_amount(satoshis: int) -> float:
    return float(np.log1p(max(0, satoshis)))


def compute_features_from_transactions(address: str, transactions: list) -> np.ndarray:
    """Compute the 12 log-scaled wallet features from raw mempool transactions."""
    total_sent = total_received = total_fees = 0
    num_send_txs = num_receive_txs = 0
    sent_amounts, received_amounts, tx_times = [], [], []

    for tx in transactions:
        if not tx.get("status", {}).get("confirmed", False):
            continue
        tx_time = tx["status"].get("block_time", 0)
        if tx_time > 0:
            tx_times.append(tx_time)

        is_sender = any(
            (inp.get("prevout") or {}).get("scriptpubkey_address") == address
            for inp in tx.get("vin", [])
        )
        received_in_tx = sum(
            out.get("value", 0)
            for out in tx.get("vout", [])
            if out.get("scriptpubkey_address") == address
        )

        if is_sender:
            num_send_txs += 1
            for out in tx.get("vout", []):
                if out.get("scriptpubkey_address") != address:
                    amount = out.get("value", 0)
                    total_sent += amount
                    if amount > 0:
                        sent_amounts.append(amount)
            total_fees += tx.get("fee", 0)

        if received_in_tx > 0:
            num_receive_txs += 1
            total_received += received_in_tx
            received_amounts.append(received_in_tx)

    # Satoshis → BTC (training data scale).
    total_sent *= SAT_TO_BTC
    total_received *= SAT_TO_BTC
    total_fees *= SAT_TO_BTC
    sent_amounts = [a * SAT_TO_BTC for a in sent_amounts]
    received_amounts = [a * SAT_TO_BTC for a in received_amounts]

    total_txs = num_send_txs + num_receive_txs
    lifetime_seconds = (max(tx_times) - min(tx_times)) if len(tx_times) >= 2 else 0
    activity_rate = np.clip((total_txs / max(lifetime_seconds, 1)) * 86400, 0, 1000)
    send_receive_ratio = np.clip(total_sent / max(total_received, 1e-10), 0, 100)
    in_out_balance = np.clip(num_receive_txs / max(num_send_txs, 1e-10), 0, 100)
    fee_per_tx = total_fees / max(total_txs, 1)

    if len(tx_times) >= 2:
        ts_sorted = sorted(tx_times)
        intervals = [ts_sorted[i + 1] - ts_sorted[i] for i in range(len(ts_sorted) - 1)]
        blocks_btwn_txs_mean = np.mean(intervals) / 600
    else:
        blocks_btwn_txs_mean = 0

    total_transacted = total_sent + total_received
    fee_share_mean = np.clip(total_fees / max(total_transacted, 1e-10), 0, 1)
    avg_tx_size = total_transacted / max(total_txs, 1)
    tx_size_range = (max(sent_amounts) - min(sent_amounts)) if sent_amounts else 0
    max_sent = max(sent_amounts) if sent_amounts else 0
    max_received = max(received_amounts) if received_amounts else 0

    features_raw = np.array([
        lifetime_seconds, activity_rate, in_out_balance, total_txs,
        send_receive_ratio, fee_per_tx, blocks_btwn_txs_mean, fee_share_mean,
        avg_tx_size, tx_size_range, max_sent, max_received,
    ], dtype=np.float32)
    return np.log1p(np.abs(features_raw)).astype(np.float32)


def _parse_transactions(center_address: str, transactions: list):
    """Extract neighbor addresses and directed edges from raw transactions."""
    neighbor_set = set()
    edge_data = []  # (neighbor_addr, direction, amount_log, ts_norm)

    for tx in transactions:
        if not tx.get("status", {}).get("confirmed", False):
            continue
        tx_time = tx["status"].get("block_time", 0)
        if tx_time == 0:
            continue
        ts_norm = _normalize_timestamp(tx_time)

        is_sender = any(
            (inp.get("prevout") or {}).get("scriptpubkey_address") == center_address
            for inp in tx.get("vin", [])
        )
        received_amount = 0
        is_receiver = False
        for out in tx.get("vout", []):
            if out.get("scriptpubkey_address") == center_address:
                is_receiver = True
                received_amount += out.get("value", 0)

        if is_sender:
            for out in tx.get("vout", []):
                recipient = out.get("scriptpubkey_address")
                amount = out.get("value", 0)
                if recipient and recipient != center_address and amount > 0:
                    neighbor_set.add(recipient)
                    edge_data.append((recipient, "out", _log_amount(amount), ts_norm))

        if is_receiver and received_amount > 0:
            for inp in tx.get("vin", []):
                sender = (inp.get("prevout") or {}).get("scriptpubkey_address")
                if sender and sender != center_address:
                    neighbor_set.add(sender)
                    edge_data.append((sender, "in", _log_amount(received_amount), ts_norm))

    return list(neighbor_set), edge_data


def build_ego_graph(address: str, transactions: list) -> Data:
    """Build a depth-1 ego-graph (PyG Data) for the wallet, center node = index 0."""
    features = compute_features_from_transactions(address, transactions)
    neighbors, edge_data = _parse_transactions(address, transactions)

    all_nodes = [address] + neighbors
    addr_to_idx = {addr: i for i, addr in enumerate(all_nodes)}
    num_nodes = len(all_nodes)

    x = np.zeros((num_nodes, NUM_NODE_FEATURES), dtype=np.float32)
    x[0] = features

    edge_index_list, edge_attr_list = [], []
    for neighbor_addr, direction, amount_log, ts_norm in edge_data:
        idx = addr_to_idx.get(neighbor_addr)
        if idx is None:
            continue
        if direction == "out":
            edge_index_list.append([0, idx])
            edge_attr_list.append([amount_log, 1.0, ts_norm])
        else:
            edge_index_list.append([idx, 0])
            edge_attr_list.append([amount_log, 0.0, ts_norm])

    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float)

    y = torch.full((num_nodes,), -1, dtype=torch.long)
    data = Data(x=torch.from_numpy(x), edge_index=edge_index,
                edge_attr=edge_attr, y=y, num_nodes=num_nodes)
    data.center_address = address
    data.num_ghost_nodes = num_nodes - 1
    data.num_edges = edge_index.size(1) if edge_index.numel() > 0 else 0
    return data


# =============================================================================
# SECTION 4 — mempool.space HTTP helpers
# =============================================================================
def _get(url: str, max_retries: int = 3, backoff: float = 5.0, timeout: int = 15):
    """GET with backoff on 429 / transient errors (mirrors the project notebook)."""
    headers = {"User-Agent": "Mozilla/5.0"}
    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=timeout, headers=headers)
            if r.status_code == 429:
                time.sleep(backoff)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            last_err = exc
            if attempt < max_retries - 1:
                time.sleep(backoff)
    raise RuntimeError(f"mempool.space request failed after {max_retries} tries: {last_err}")


def fetch_address_stats(address: str) -> dict:
    return _get(f"{MEMPOOL_API}/address/{address}")


def fetch_transactions(address: str, max_pages: int = 2, sleep_between: float = 0.5) -> list:
    """Paginate /address/{addr}/txs. Cap ~ max_pages * ~50 txs to keep runtime short."""
    first = _get(f"{MEMPOOL_API}/address/{address}/txs")
    if not first:
        return []
    all_txs = list(first)
    last_id = first[-1].get("txid")
    for _ in range(max_pages - 1):
        if not last_id:
            break
        time.sleep(sleep_between)
        try:
            page = _get(f"{MEMPOOL_API}/address/{address}/txs/chain/{last_id}")
        except RuntimeError:
            break
        if not page:
            break
        all_txs.extend(page)
        last_id = page[-1].get("txid")
        if len(page) < 25:
            break
    return all_txs


def fetch_btc_usd_price() -> float | None:
    """Current BTC→USD price, or None if the price feed is unavailable."""
    try:
        return float(_get(f"{MEMPOOL_API}/v1/prices").get("USD"))
    except Exception:
        return None


def is_valid_bitcoin_address(addr: str) -> bool:
    """Accept legacy P2PKH/P2SH (1.../3...) and Bech32 (bc1...) formats."""
    return bool(
        re.match(r"^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$", addr)
        or re.match(r"^bc1[a-z0-9]{39,59}$", addr)
    )


# =============================================================================
# SECTION 5 — Model loading (local first, else download)
# =============================================================================
def _torch_load(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # older torch without weights_only
        return torch.load(path, map_location=device)


def _download(url: str, dest: Path) -> None:
    print(f"  → downloading {dest.name} ...", flush=True)
    with requests.get(url, stream=True, timeout=60,
                      headers={"User-Agent": "Mozilla/5.0"}) as r:
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 16):
                fh.write(chunk)


def _resolve_model_files() -> tuple[Path, Path | None]:
    """Find gnn_model.pt (+ temperature.pt): repo-local first, then cached download."""
    here = Path(__file__).resolve()

    # 1. Inside the project repo (walk up looking for outputs/gnn_model.pt).
    for parent in [here.parent, *here.parents]:
        candidate = parent / "outputs" / "gnn_model.pt"
        if candidate.exists():
            temp = parent / "outputs" / "temperature.pt"
            return candidate, (temp if temp.exists() else None)

    # 2. Previously downloaded cache next to this script.
    cache = here.parent / ".model_cache"
    model, temp = cache / "gnn_model.pt", cache / "temperature.pt"
    if model.exists():
        return model, (temp if temp.exists() else None)

    # 3. Download from GitHub.
    print("  Model not found locally — fetching from GitHub ...")
    _download(MODEL_URL, model)
    try:
        _download(TEMPERATURE_URL, temp)
    except Exception:
        temp = None
    return model, (temp if temp.exists() else None)


def load_model():
    model_path, temp_path = _resolve_model_files()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OptimalBitcoinGNN(
        num_node_features=NUM_NODE_FEATURES,
        num_edge_features=NUM_EDGE_FEATURES,
        hidden_dim=64,
    ).to(device)
    model.load_state_dict(_torch_load(model_path, device))
    model.eval()

    temperature = 1.0
    if temp_path is not None:
        temperature = float(_torch_load(temp_path, device)["temperature"])
    return model, temperature, device


# =============================================================================
# SECTION 6 — Inference: classification + feature importance
# =============================================================================
def classify(model, graph, temperature, device) -> dict:
    x = graph.x.to(device)
    edge_index = graph.edge_index.to(device)
    edge_attr = graph.edge_attr.to(device) if graph.edge_attr.numel() > 0 else None
    batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x, edge_index, edge_attr, batch)[0]
        probs = F.softmax(logits / temperature, dim=0).cpu().numpy()
    prob_benign, prob_criminal = float(probs[0]), float(probs[1])
    return {
        "prob_benign": prob_benign,
        "prob_criminal": prob_criminal,
        "risk_score": prob_criminal,
        "classification": "criminal" if prob_criminal > 0.5 else "benign",
        "confidence": abs(prob_criminal - 0.5) * 2,
    }


def feature_importance(model, graph, device) -> dict:
    """Gradient saliency of the criminal-class logit w.r.t. center-node features."""
    x = graph.x.clone().detach().to(device).requires_grad_(True)
    edge_index = graph.edge_index.to(device)
    edge_attr = graph.edge_attr.to(device) if graph.edge_attr.numel() > 0 else None
    batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
    out = model(x, edge_index, edge_attr, batch)
    out[0, 1].backward()
    grads = x.grad[0].abs().cpu().numpy()
    total = grads.sum()
    if total > 0:
        grads = grads / total
    return dict(zip(FEATURE_COLUMNS, grads.tolist()))


# =============================================================================
# SECTION 7 — Report assembly + rendering
# =============================================================================
def summarize_money(stats: dict) -> dict:
    """Accurate on-chain totals from address stats (funded/spent txo sums)."""
    chain = stats.get("chain_stats", {})
    mp = stats.get("mempool_stats", {})
    funded = (chain.get("funded_txo_sum", 0) + mp.get("funded_txo_sum", 0)) * SAT_TO_BTC
    spent = (chain.get("spent_txo_sum", 0) + mp.get("spent_txo_sum", 0)) * SAT_TO_BTC
    tx_count = chain.get("tx_count", 0) + mp.get("tx_count", 0)
    return {
        "total_received_btc": funded,
        "total_sent_btc": spent,
        "balance_btc": funded - spent,
        "total_tx_count": tx_count,
    }


def build_report(address: str, stats: dict, graph, verdict: dict,
                 importance: dict, fetched_txs: int, btc_usd: float | None) -> dict:
    money = summarize_money(stats)
    top3 = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:3]
    return {
        "wallet_address": address,
        "analyzed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "verdict": verdict["classification"],
        "risk_percentage": round(verdict["risk_score"] * 100, 2),
        "confidence_percentage": round(verdict["confidence"] * 100, 2),
        "money": money,
        "btc_usd_price": btc_usd,
        "top_features": [
            {"feature": name, "label": FEATURE_DISPLAY.get(name, name),
             "importance_percentage": round(score * 100, 2)}
            for name, score in top3
        ],
        "graph": {
            "nodes": int(graph.num_nodes),
            "edges": int(graph.num_edges),
            "neighbors": int(graph.num_ghost_nodes),
            "transactions_analyzed": fetched_txs,
        },
    }


def _usd(btc: float, price: float | None) -> str:
    return f"  (${btc * price:,.2f})" if price else ""


def render_report(report: dict) -> str:
    money = report["money"]
    price = report["btc_usd_price"]
    is_criminal = report["verdict"] == "criminal"
    badge = "⚠  CRIMINAL" if is_criminal else "✓  BENIGN"

    lines = []
    add = lines.append
    W = 64
    add("=" * W)
    add("  BITCOIN WALLET RISK REPORT".ljust(W))
    add("=" * W)
    add(f"  Wallet    : {report['wallet_address']}")
    add(f"  Analyzed  : {report['analyzed_at_utc']}")
    add("")
    add(f"  VERDICT   : {badge}")
    add(f"  Risk      : {report['risk_percentage']:.1f}%  "
        f"chance this wallet is criminal")
    add(f"  Confidence: {report['confidence_percentage']:.1f}%")
    add("")
    add("  " + "-" * (W - 4))
    add("  MONEY")
    add(f"    Money in  (total received) : {money['total_received_btc']:.8f} BTC"
        f"{_usd(money['total_received_btc'], price)}")
    add(f"    Money out (total sent)     : {money['total_sent_btc']:.8f} BTC"
        f"{_usd(money['total_sent_btc'], price)}")
    add(f"    Current balance            : {money['balance_btc']:.8f} BTC"
        f"{_usd(money['balance_btc'], price)}")
    add(f"    Total transactions         : {money['total_tx_count']:,}")
    add("")
    add("  " + "-" * (W - 4))
    add("  TOP 3 FEATURES DRIVING THIS DECISION")
    for i, feat in enumerate(report["top_features"], 1):
        add(f"    {i}. {feat['label']}")
        add(f"       contribution: {feat['importance_percentage']:.1f}%")
    add("")
    add("  " + "-" * (W - 4))
    g = report["graph"]
    add(f"  Analyzed graph: {g['nodes']} nodes · {g['edges']} edges · "
        f"{g['neighbors']} neighbors")
    if g["transactions_analyzed"] < money["total_tx_count"]:
        add(f"  (based on {g['transactions_analyzed']} of "
            f"{money['total_tx_count']:,} on-chain txs — capped for speed)")
    add("=" * W)
    return "\n".join(lines)


# =============================================================================
# SECTION 8 — Orchestration / CLI
# =============================================================================
def analyze(address: str, model_bundle=None) -> dict:
    if not is_valid_bitcoin_address(address):
        raise ValueError(f"Invalid Bitcoin address format: {address!r}")

    if model_bundle is None:
        print("Loading model ...")
        model_bundle = load_model()
    model, temperature, device = model_bundle

    print(f"Fetching wallet data for {address} ...")
    stats = fetch_address_stats(address)
    txs = fetch_transactions(address, max_pages=2)
    if not txs:
        raise RuntimeError("This wallet has no transactions; classification is not meaningful.")

    print(f"Building ego-graph from {len(txs)} transactions ...")
    graph = build_ego_graph(address, txs)

    print("Running the model ...")
    verdict = classify(model, graph, temperature, device)
    importance = feature_importance(model, graph, device)
    btc_usd = fetch_btc_usd_price()

    return build_report(address, stats, graph, verdict, importance, len(txs), btc_usd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a Bitcoin wallet's criminal-risk with a trained GNN.")
    parser.add_argument("address", nargs="?", help="Bitcoin wallet address to analyze")
    parser.add_argument("--json", action="store_true", help="print raw JSON instead of a report")
    args = parser.parse_args()

    model_bundle = None      # loaded once, reused for every wallet
    pending = args.address   # analyze the CLI-provided address first, if any
    first = True

    try:
        while True:
            if pending:
                address, pending = pending, None
            else:
                prompt = "Enter a Bitcoin wallet address" if first else "Check another wallet"
                address = input(f"\n{prompt} (press Ctrl+C to close): ").strip()
            first = False

            if not address:
                continue

            if not is_valid_bitcoin_address(address):
                print(f"[!] Invalid Bitcoin address format: {address!r}")
                continue

            if model_bundle is None:
                print("Loading model ...")
                model_bundle = load_model()

            try:
                report = analyze(address, model_bundle)
            except (ValueError, RuntimeError) as exc:
                print(f"[!] {exc}")
                continue
            except requests.RequestException as exc:
                print(f"[!] Network error talking to mempool.space: {exc}")
                continue

            print()
            if args.json:
                print(json.dumps(report, indent=2))
            else:
                print(render_report(report))
    except (EOFError, KeyboardInterrupt):
        print("\nClosing. Goodbye.")


if __name__ == "__main__":
    main()
