#!/usr/bin/env python3
"""
Bitcoin Wallet Risk Analyzer — Standalone CLI
==============================================
A self-contained tool that classifies Bitcoin wallets as benign or criminal
using a Graph Neural Network (GATv2). No local project imports required.

Usage:
    python wallet_analyzer.py [ADDRESS] [--model PATH] [--output-dir DIR] [--no-charts] [--open-charts]

If no address is given, prompts interactively (double-click friendly).
Internet required (mempool.space API).
"""

import argparse
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch Geometric imports
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, LayerNorm

# Matplotlib — use non-interactive backend so it works headless / in exe
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# =============================================================================
# CONSTANTS (inlined from src/graph/config.py)
# =============================================================================

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

FEATURE_DISPLAY_NAMES = [
    "lifetime seconds",
    "activity rate",
    "in/out balance",
    "total txs",
    "send/receive ratio",
    "fee per tx",
    "blocks btwn txs mean",
    "fee share mean",
    "avg tx size",
    "tx size range",
    "max sent",
    "max received",
]

NUM_NODE_FEATURES = 12  # hardcoded to match trained model weights
NUM_EDGE_FEATURES = 3   # amount_log, direction, timestamp_norm

TIMESTAMP_MIN = 1231006505   # Bitcoin genesis block: 2009-01-03
TIMESTAMP_MAX = 1893456000   # ~2030-01-01

MEMPOOL_API = "https://mempool.space/api"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# =============================================================================
# COLOR PALETTE (matching frontend)
# =============================================================================

CLR_PURPLE = "#8b5cf6"
CLR_GREEN = "#22c55e"
CLR_RED = "#ef4444"
CLR_ORANGE = "#f59e0b"
CLR_BLUE = "#3b82f6"
CLR_GRAY_200 = "#e5e7eb"
CLR_GRAY_500 = "#6b7280"
CLR_GRAY_700 = "#374151"


# =============================================================================
# MODEL — OptimalBitcoinGNN (inlined from src/models/optimal_gnn.py)
# =============================================================================

class OptimalBitcoinGNN(nn.Module):
    """3-layer GATv2Conv with residual connections for Bitcoin wallet classification."""

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
    ):
        super().__init__()
        self.num_node_features = num_node_features
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Layer 1: 12 -> 64*4 = 256
        self.conv1 = GATv2Conv(
            in_channels=num_node_features,
            out_channels=hidden_dim,
            heads=num_heads_1,
            edge_dim=num_edge_features,
            concat=True,
            dropout=dropout,
        )
        self.norm1 = LayerNorm(hidden_dim * num_heads_1)

        # Layer 2: 256 -> 64*4 = 256
        self.conv2 = GATv2Conv(
            in_channels=hidden_dim * num_heads_1,
            out_channels=hidden_dim,
            heads=num_heads_2,
            edge_dim=num_edge_features,
            concat=True,
            dropout=dropout,
        )
        self.norm2 = LayerNorm(hidden_dim * num_heads_2)

        # Residual projection
        self.residual_proj = nn.Linear(
            hidden_dim * num_heads_1,
            hidden_dim * num_heads_2,
        )

        # Layer 3: 256 -> 32*2 = 64
        self.conv3 = GATv2Conv(
            in_channels=hidden_dim * num_heads_2,
            out_channels=hidden_dim // 2,
            heads=num_heads_3,
            edge_dim=num_edge_features,
            concat=True,
            dropout=dropout * 0.5,
        )
        self.norm3 = LayerNorm((hidden_dim // 2) * num_heads_3)

        # Classifier: 64 -> 32 -> 2
        classifier_input_dim = (hidden_dim // 2) * num_heads_3
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(final_dropout),
            nn.Linear(hidden_dim // 2, 2),
        )
        self.dropout_layer = nn.Dropout(dropout)
        self.dropout_light = nn.Dropout(dropout * 0.5)

    def forward(self, x, edge_index, edge_attr, batch):
        # Layer 1
        h = self.conv1(x, edge_index, edge_attr=edge_attr)
        h = self.norm1(h)
        h = F.elu(h)
        h = self.dropout_layer(h)

        h_residual = h

        # Layer 2
        h = self.conv2(h, edge_index, edge_attr=edge_attr)
        h = self.norm2(h)
        h = F.elu(h)
        h = self.dropout_layer(h)

        # Residual connection
        h = h + self.residual_proj(h_residual)

        # Layer 3
        h = self.conv3(h, edge_index, edge_attr=edge_attr)
        h = self.norm3(h)
        h = F.elu(h)
        h = self.dropout_light(h)

        # Extract center node embeddings
        h = self._get_center_embeddings(h, batch)

        return self.classifier(h)

    def _get_center_embeddings(self, h, batch):
        batch_size = batch.max().item() + 1
        center_embeddings = []
        for i in range(batch_size):
            mask = batch == i
            graph_nodes = h[mask]
            center_embeddings.append(graph_nodes[0])
        return torch.stack(center_embeddings)


# =============================================================================
# GRAPH BUILDER — EgoGraphBuilder (inlined from src/graph/graph_builder.py)
# =============================================================================

class EgoGraphBuilder:
    """Builds ego-graphs (depth=1) for wallet addresses."""

    def __init__(self):
        self.num_features = NUM_NODE_FEATURES

    def _normalize_timestamp(self, ts: int) -> float:
        if ts <= TIMESTAMP_MIN:
            return 0.0
        if ts >= TIMESTAMP_MAX:
            return 1.0
        return (ts - TIMESTAMP_MIN) / (TIMESTAMP_MAX - TIMESTAMP_MIN)

    def _log_amount(self, satoshis: int) -> float:
        return np.log1p(max(0, satoshis))

    def _parse_transactions(
        self, center_address: str, transactions: List[Dict]
    ) -> Tuple[List[str], List[Tuple]]:
        neighbor_set: set = set()
        edge_data: list = []

        for tx in transactions:
            if not tx.get("status", {}).get("confirmed", False):
                continue
            tx_time = tx["status"].get("block_time", 0)
            if tx_time == 0:
                continue
            ts_norm = self._normalize_timestamp(tx_time)

            # Is center a sender?
            is_sender = False
            for inp in tx.get("vin", []):
                prevout = inp.get("prevout") or {}
                if prevout.get("scriptpubkey_address") == center_address:
                    is_sender = True
                    break

            # Is center a receiver?
            is_receiver = False
            received_amount = 0
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
                        edge_data.append((recipient, "out", self._log_amount(amount), ts_norm))

            if is_receiver and received_amount > 0:
                for inp in tx.get("vin", []):
                    prevout = inp.get("prevout") or {}
                    sender = prevout.get("scriptpubkey_address")
                    if sender and sender != center_address:
                        neighbor_set.add(sender)
                        edge_data.append((sender, "in", self._log_amount(received_amount), ts_norm))

        return list(neighbor_set), edge_data

    def build_ego_graph(
        self,
        center_address: str,
        center_features: np.ndarray,
        center_label: int,
        transactions: List[Dict],
    ) -> Data:
        neighbors, edge_data = self._parse_transactions(center_address, transactions)
        all_nodes = [center_address] + neighbors
        addr_to_idx = {addr: i for i, addr in enumerate(all_nodes)}
        num_nodes = len(all_nodes)

        x = np.zeros((num_nodes, self.num_features), dtype=np.float32)
        x[0] = center_features

        if edge_data:
            edge_index_list = []
            edge_attr_list = []
            for neighbor_addr, direction, amount_log, ts_norm in edge_data:
                neighbor_idx = addr_to_idx.get(neighbor_addr)
                if neighbor_idx is None:
                    continue
                if direction == "out":
                    edge_index_list.append([0, neighbor_idx])
                    edge_attr_list.append([amount_log, 1.0, ts_norm])
                else:
                    edge_index_list.append([neighbor_idx, 0])
                    edge_attr_list.append([amount_log, 0.0, ts_norm])

            if edge_index_list:
                edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 3), dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 3), dtype=torch.float)

        y = torch.full((num_nodes,), -1, dtype=torch.long)
        y[0] = center_label

        data = Data(
            x=torch.from_numpy(x),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_nodes=num_nodes,
        )
        data.center_address = center_address
        data.num_ghost_nodes = num_nodes - 1
        data.num_edges = edge_index.size(1) if edge_index.numel() > 0 else 0
        return data

    def build_empty_graph(
        self,
        center_address: str,
        center_features: np.ndarray,
        center_label: int,
    ) -> Data:
        x = torch.from_numpy(center_features.reshape(1, -1).astype(np.float32))
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float)
        y = torch.tensor([center_label], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=1)
        data.center_address = center_address
        data.num_ghost_nodes = 0
        data.num_edges = 0
        return data

    def compute_features_from_transactions(
        self, address: str, transactions: List[Dict]
    ) -> np.ndarray:
        total_sent = 0
        total_received = 0
        total_fees = 0
        num_send_txs = 0
        num_receive_txs = 0
        sent_amounts: list = []
        received_amounts: list = []
        tx_times: list = []

        for tx in transactions:
            if not tx.get("status", {}).get("confirmed", False):
                continue
            tx_time = tx["status"].get("block_time", 0)
            if tx_time > 0:
                tx_times.append(tx_time)

            is_sender = False
            for inp in tx.get("vin", []):
                prevout = inp.get("prevout") or {}
                if prevout.get("scriptpubkey_address") == address:
                    is_sender = True
                    break

            received_in_tx = 0
            for out in tx.get("vout", []):
                if out.get("scriptpubkey_address") == address:
                    received_in_tx += out.get("value", 0)

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

        total_txs = num_send_txs + num_receive_txs
        lifetime_seconds = (max(tx_times) - min(tx_times)) if len(tx_times) >= 2 else 0
        activity_rate = total_txs / max(lifetime_seconds, 1)
        send_receive_ratio = total_sent / max(total_received, 1)
        in_out_balance = num_receive_txs / max(num_send_txs, 1)
        fee_per_tx = total_fees / max(total_txs, 1)

        if len(tx_times) >= 2:
            tx_times_sorted = sorted(tx_times)
            intervals = [tx_times_sorted[i + 1] - tx_times_sorted[i] for i in range(len(tx_times_sorted) - 1)]
            blocks_btwn_txs_mean = np.mean(intervals) / 600
        else:
            blocks_btwn_txs_mean = 0

        total_transacted = total_sent + total_received
        fee_share_mean = total_fees / max(total_transacted, 1)
        avg_tx_size = total_transacted / max(total_txs, 1)

        all_amounts = sent_amounts + received_amounts
        tx_size_range = (max(all_amounts) - min(all_amounts)) if all_amounts else 0

        max_sent = max(sent_amounts) if sent_amounts else 0
        max_received = max(received_amounts) if received_amounts else 0

        features_raw = np.array(
            [
                lifetime_seconds,
                activity_rate,
                in_out_balance,
                total_txs,
                send_receive_ratio,
                fee_per_tx,
                blocks_btwn_txs_mean,
                fee_share_mean,
                avg_tx_size,
                tx_size_range,
                max_sent,
                max_received,
            ],
            dtype=np.float32,
        )
        return np.log1p(np.abs(features_raw)).astype(np.float32)

    def build_graph_for_new_address(
        self, address: str, transactions: List[Dict], label: int = -1
    ) -> Data:
        features = self.compute_features_from_transactions(address, transactions)
        if transactions:
            return self.build_ego_graph(
                center_address=address,
                center_features=features,
                center_label=label,
                transactions=transactions,
            )
        else:
            return self.build_empty_graph(
                center_address=address,
                center_features=features,
                center_label=label,
            )


# =============================================================================
# PIPELINE FUNCTIONS
# =============================================================================


MODEL_DOWNLOAD_URL = (
    "https://github.com/NehorayChalfon0166/final_project/releases/download/v1.0.0/gnn_model.pt"
)


def download_model(dest_path: str) -> str:
    """Download the trained GNN model from GitHub Releases."""
    print(f"\n  Model not found locally. Downloading from GitHub Releases...")
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    resp = requests.get(MODEL_DOWNLOAD_URL, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 // total
                print(f"\r  Downloading model... {pct}%", end="", flush=True)
    print(f"\r  Model downloaded ({downloaded // 1024} KB)           ")
    return dest_path


def get_model_path(model_arg: str | None) -> str:
    """Resolve the model path — works both as a script and as a frozen PyInstaller exe."""
    candidates: list[str] = []

    if model_arg:
        candidates.append(model_arg)
        candidates.append(os.path.abspath(model_arg))

    # When running as a PyInstaller bundle, _MEIPASS is the temp extract dir
    if getattr(sys, "_MEIPASS", None):
        candidates.append(os.path.join(sys._MEIPASS, "gnn_model.pt"))

    # Relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(script_dir, "outputs", "gnn_model.pt"))
    candidates.append(os.path.join(script_dir, "..", "outputs", "gnn_model.pt"))
    candidates.append(os.path.join(script_dir, "outputs", "gnn_checkpoint.pt"))
    candidates.append(os.path.join(script_dir, "..", "outputs", "gnn_checkpoint.pt"))

    for path in candidates:
        if os.path.isfile(path):
            return path

    # Model not found locally — auto-download to standalone/outputs/
    default_path = os.path.join(script_dir, "outputs", "gnn_model.pt")
    return download_model(default_path)


def fetch_transactions(address: str) -> List[Dict]:
    """Fetch confirmed transactions from mempool.space API."""
    url = f"{MEMPOOL_API}/address/{address}/txs"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 400:
            raise ValueError(f"Invalid Bitcoin address: {address}")
        else:
            raise ConnectionError(f"mempool.space returned HTTP {r.status_code}")
    except requests.exceptions.ConnectionError:
        raise ConnectionError("No internet connection. mempool.space API is required.")
    except requests.exceptions.Timeout:
        raise ConnectionError("Request timed out. Please try again.")


def fetch_balance(address: str) -> Dict:
    """Fetch address balance & stats from mempool.space API."""
    url = f"{MEMPOOL_API}/address/{address}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return {}
        data = r.json()
        chain = data.get("chain_stats", {})
        mempool = data.get("mempool_stats", {})

        funded_sats = chain.get("funded_txo_sum", 0) + mempool.get("funded_txo_sum", 0)
        spent_sats = chain.get("spent_txo_sum", 0) + mempool.get("spent_txo_sum", 0)
        balance_sats = funded_sats - spent_sats

        return {
            "balance_sats": balance_sats,
            "balance_btc": balance_sats / 1e8,
            "total_received_sats": funded_sats,
            "total_received_btc": funded_sats / 1e8,
            "total_sent_sats": spent_sats,
            "total_sent_btc": spent_sats / 1e8,
        }
    except Exception:
        return {}


def run_inference(graph_data: Data, model_path: str) -> Dict:
    """Run GNN inference on a single graph, return classification results."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = OptimalBitcoinGNN(
        num_node_features=NUM_NODE_FEATURES,
        num_edge_features=NUM_EDGE_FEATURES,
    )
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    with torch.no_grad():
        x = graph_data.x.to(device)
        edge_index = graph_data.edge_index.to(device)
        edge_attr = graph_data.edge_attr.to(device) if graph_data.edge_attr.numel() > 0 else None
        batch = torch.zeros(graph_data.num_nodes, dtype=torch.long, device=device)

        output = model(x, edge_index, edge_attr, batch)
        logits = output[0].cpu()
        probabilities = F.softmax(logits, dim=0).numpy()

        prob_benign = float(probabilities[0])
        prob_criminal = float(probabilities[1])
        risk_score = prob_criminal
        classification = "criminal" if risk_score > 0.5 else "benign"
        confidence = abs(risk_score - 0.5) * 2

    return {
        "classification": classification,
        "risk_score": risk_score,
        "confidence": confidence,
        "prob_benign": prob_benign,
        "prob_criminal": prob_criminal,
    }


def compute_feature_importance(graph_data: Data, model_path: str) -> Dict[str, float]:
    """Gradient-based feature importance. Returns dict {display_name: percentage}."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = OptimalBitcoinGNN(
        num_node_features=NUM_NODE_FEATURES,
        num_edge_features=NUM_EDGE_FEATURES,
    )
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    x = graph_data.x.clone().requires_grad_(True).to(device)
    edge_index = graph_data.edge_index.to(device)
    edge_attr = graph_data.edge_attr.to(device)
    batch = torch.zeros(graph_data.num_nodes, dtype=torch.long, device=device)

    output = model(x, edge_index, edge_attr, batch)
    target_output = output[0, 1]  # criminal class logit
    target_output.backward()

    gradients = x.grad[0].abs().cpu().numpy()
    total = gradients.sum()
    if total > 0:
        gradients = gradients / total

    importance = {}
    for i, name in enumerate(FEATURE_DISPLAY_NAMES):
        importance[name] = float(gradients[i]) * 100  # as percentage
    return importance


# =============================================================================
# CHART FUNCTIONS (matching frontend visual style)
# =============================================================================


def plot_feature_importance(importance: Dict[str, float], output_dir: str) -> str:
    """Horizontal bar chart of feature importance (purple #8b5cf6)."""
    sorted_items = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, color=CLR_PURPLE, edgecolor="none", height=0.6)

    # Round right corners visually
    for bar in bars:
        bar.set_capstyle("round")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=11, color=CLR_GRAY_700)
    ax.invert_yaxis()
    ax.set_xlabel("Importance (%)", fontsize=12, color=CLR_GRAY_500)
    ax.set_title("Feature Importance", fontsize=16, fontweight="bold", color=CLR_GRAY_700, pad=15)
    ax.tick_params(axis="x", colors=CLR_GRAY_500, labelsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.3, color=CLR_GRAY_200)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(CLR_GRAY_200)
    ax.spines["bottom"].set_color(CLR_GRAY_200)

    # Value labels on bars
    for i, (v, bar) in enumerate(zip(values, bars)):
        ax.text(v + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", fontsize=9, color=CLR_GRAY_500)

    plt.tight_layout()
    path = os.path.join(output_dir, "feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_tx_volume(transactions: List[Dict], address: str, output_dir: str) -> str:
    """Weekly incoming/outgoing bar chart (green/red)."""
    # Aggregate into weekly buckets
    weekly: Dict[str, Dict[str, float]] = {}

    for tx in transactions:
        if not tx.get("status", {}).get("confirmed", False):
            continue
        tx_time = tx["status"].get("block_time", 0)
        if tx_time == 0:
            continue

        # Week key
        from datetime import datetime, timedelta, timezone
        dt = datetime.fromtimestamp(tx_time, tz=timezone.utc)
        # Start of week (Monday)
        week_start = dt - timedelta(days=dt.weekday())
        week_key = week_start.strftime("%Y-%m-%d")

        if week_key not in weekly:
            weekly[week_key] = {"incoming": 0.0, "outgoing": 0.0}

        # Check if sender
        is_sender = False
        for inp in tx.get("vin", []):
            prevout = inp.get("prevout") or {}
            if prevout.get("scriptpubkey_address") == address:
                is_sender = True
                break

        # Incoming
        received = 0
        for out in tx.get("vout", []):
            if out.get("scriptpubkey_address") == address:
                received += out.get("value", 0)

        if received > 0:
            weekly[week_key]["incoming"] += received / 1e8

        if is_sender:
            sent = 0
            for out in tx.get("vout", []):
                if out.get("scriptpubkey_address") != address:
                    sent += out.get("value", 0)
            weekly[week_key]["outgoing"] += sent / 1e8

    if not weekly:
        return ""

    # Sort by date
    sorted_weeks = sorted(weekly.keys())
    labels = sorted_weeks
    incoming = [weekly[w]["incoming"] for w in sorted_weeks]
    outgoing = [weekly[w]["outgoing"] for w in sorted_weeks]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("white")

    x = np.arange(len(labels))
    width = 0.35

    bars_in = ax.bar(x - width / 2, incoming, width, label="Received", color=CLR_GREEN,
                     edgecolor="none")
    bars_out = ax.bar(x + width / 2, outgoing, width, label="Sent", color=CLR_RED,
                      edgecolor="none")

    ax.set_ylabel("BTC", fontsize=12, color=CLR_GRAY_500)
    ax.set_title("Transaction Volume (Weekly)", fontsize=16, fontweight="bold",
                 color=CLR_GRAY_700, pad=15)

    # Show subset of x labels if too many
    if len(labels) > 20:
        step = max(1, len(labels) // 10)
        tick_indices = list(range(0, len(labels), step))
        ax.set_xticks([x[i] for i in tick_indices])
        ax.set_xticklabels([labels[i] for i in tick_indices], rotation=45, ha="right",
                           fontsize=9, color=CLR_GRAY_500)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9, color=CLR_GRAY_500)

    ax.tick_params(axis="y", colors=CLR_GRAY_500, labelsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3, color=CLR_GRAY_200)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(CLR_GRAY_200)
    ax.spines["bottom"].set_color(CLR_GRAY_200)
    ax.legend(fontsize=11, frameon=False)

    # Summary text
    total_in = sum(incoming)
    total_out = sum(outgoing)
    ax.text(0.5, -0.22,
            f"Total Received: {total_in:.8f} BTC  |  Total Sent: {total_out:.8f} BTC",
            transform=ax.transAxes, ha="center", fontsize=10, color=CLR_GRAY_500)

    plt.tight_layout()
    path = os.path.join(output_dir, "tx_volume.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_risk_gauge(risk_score: float, classification: str, output_dir: str) -> str:
    """Semi-circle gauge with needle showing risk score."""
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("white")

    # Draw gauge arcs
    n_segments = 100
    for i in range(n_segments):
        angle_start = 180 - (i / n_segments) * 180
        angle_end = 180 - ((i + 1) / n_segments) * 180
        ratio = i / n_segments
        # Color gradient: green -> yellow -> orange -> red
        if ratio < 0.25:
            r, g, b = 0.13, 0.77, 0.37  # green
        elif ratio < 0.5:
            t = (ratio - 0.25) / 0.25
            r = 0.13 + t * (0.96 - 0.13)
            g = 0.77 + t * (0.62 - 0.77)
            b = 0.37 + t * (0.04 - 0.37)
        elif ratio < 0.75:
            t = (ratio - 0.5) / 0.25
            r = 0.96 + t * (0.94 - 0.96)
            g = 0.62 + t * (0.27 - 0.62)
            b = 0.04 + t * (0.27 - 0.04)
        else:
            r, g, b = 0.94, 0.27, 0.27  # red

        wedge = mpatches.Wedge(
            center=(0, 0), r=1.0, theta1=angle_end, theta2=angle_start,
            facecolor=(r, g, b), edgecolor="white", linewidth=0.5,
        )
        ax.add_patch(wedge)

    # Inner white circle to make it a donut
    inner = plt.Circle((0, 0), 0.6, fc="white", ec="none", zorder=2)
    ax.add_patch(inner)

    # Needle
    needle_angle = 180 - risk_score * 180  # 0% = left (green), 100% = right (red)
    needle_rad = math.radians(needle_angle)
    needle_x = 0.85 * math.cos(needle_rad)
    needle_y = 0.85 * math.sin(needle_rad)
    ax.plot([0, needle_x], [0, needle_y], color=CLR_GRAY_700, linewidth=2.5, zorder=3)
    ax.plot(0, 0, "o", color=CLR_GRAY_700, markersize=8, zorder=4)

    # Score text in center
    risk_pct = risk_score * 100
    if risk_pct <= 25:
        level = "LOW"
        level_color = CLR_GREEN
    elif risk_pct <= 50:
        level = "MODERATE"
        level_color = CLR_ORANGE
    elif risk_pct <= 75:
        level = "HIGH"
        level_color = CLR_RED
    else:
        level = "CRITICAL"
        level_color = "#991b1b"

    ax.text(0, -0.05, f"{risk_pct:.1f}%", ha="center", va="center",
            fontsize=28, fontweight="bold", color=CLR_GRAY_700, zorder=5)
    ax.text(0, -0.25, level, ha="center", va="center",
            fontsize=14, fontweight="bold", color=level_color, zorder=5)

    # Labels
    ax.text(-1.05, -0.08, "0%", ha="center", fontsize=9, color=CLR_GRAY_500)
    ax.text(1.05, -0.08, "100%", ha="center", fontsize=9, color=CLR_GRAY_500)

    ax.set_title("Risk Score", fontsize=16, fontweight="bold", color=CLR_GRAY_700, pad=15)

    # Classification badge
    badge_color = CLR_RED if classification == "criminal" else CLR_GREEN
    ax.text(0, -0.55, classification.upper(), ha="center", va="center",
            fontsize=13, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=badge_color, edgecolor="none"))

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.75, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    path = os.path.join(output_dir, "risk_gauge.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# =============================================================================
# TERMINAL OUTPUT
# =============================================================================

SEPARATOR = "=" * 56


def print_header():
    print(f"\n{SEPARATOR}")
    print("  BITCOIN WALLET RISK ANALYZER")
    print(SEPARATOR)


def print_results(
    address: str,
    inference: Dict,
    balance: Dict,
    importance: Dict[str, float],
    graph_data: Data,
    num_txs: int,
    chart_paths: List[str],
):
    """Formatted ASCII output with classification, stats, balance, top features."""

    risk_pct = inference["risk_score"] * 100
    if risk_pct <= 25:
        level = "LOW"
    elif risk_pct <= 50:
        level = "MODERATE"
    elif risk_pct <= 75:
        level = "HIGH"
    else:
        level = "CRITICAL"

    print()
    print(f"  Classification : {inference['classification'].upper()}")
    print(f"  Risk Score     : {risk_pct:.1f}%  ({level})")
    print(f"  Confidence     : {inference['confidence'] * 100:.1f}%")
    print()

    if balance:
        print(f"  Balance        : {balance.get('balance_btc', 0):.8f} BTC")
        print(f"  Total Received : {balance.get('total_received_btc', 0):.8f} BTC")
        print(f"  Total Sent     : {balance.get('total_sent_btc', 0):.8f} BTC")
        print()

    # Top features
    sorted_feats = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
    max_val = sorted_feats[0][1] if sorted_feats else 1
    max_name_len = max(len(name) for name, _ in sorted_feats[:6])

    print("  Top Features:")
    for i, (name, val) in enumerate(sorted_feats[:6], 1):
        bar_len = int((val / max(max_val, 0.01)) * 20)
        bar = "#" * bar_len
        print(f"  {i}. {name:<{max_name_len}}  {val:>5.1f}%  {bar}")

    print()
    if chart_paths:
        chart_dir = os.path.dirname(chart_paths[0]) if chart_paths[0] else ""
        chart_names = [os.path.basename(p) for p in chart_paths if p]
        print(f"  Charts saved to: {chart_dir}/")
        for name in chart_names:
            print(f"    - {name}")

    print(SEPARATOR)


# =============================================================================
# MAIN
# =============================================================================


def analyze_address(address: str, args) -> None:
    """Run full analysis pipeline for a single address."""
    print(f"  Wallet: {address}\n")

    # ------------------------------------------------------------------
    # Step 1: Fetch transactions
    # ------------------------------------------------------------------
    try:
        sys.stdout.write("[1/4] Fetching transactions...        ")
        sys.stdout.flush()
        transactions = fetch_transactions(address)
        num_txs = len(transactions)
        print(f"Found {num_txs} transactions")
    except ValueError as e:
        print(f"\n  Error: {e}")
        return
    except ConnectionError as e:
        print(f"\n  Error: {e}")
        return

    if num_txs == 0:
        print("  No transactions found for this address. Cannot classify.")
        return

    # ------------------------------------------------------------------
    # Step 2: Build ego-graph
    # ------------------------------------------------------------------
    sys.stdout.write("[2/4] Building ego-graph...            ")
    sys.stdout.flush()
    builder = EgoGraphBuilder()
    graph_data = builder.build_graph_for_new_address(
        address=address, transactions=transactions, label=-1
    )
    print(f"{graph_data.num_nodes} nodes, {graph_data.num_edges} edges")

    # ------------------------------------------------------------------
    # Step 3: Run GNN inference
    # ------------------------------------------------------------------
    sys.stdout.write("[3/4] Running GNN inference...         ")
    sys.stdout.flush()
    try:
        model_path = get_model_path(args.model)
        inference = run_inference(graph_data, model_path)
        print("Done")
    except FileNotFoundError as e:
        print(f"\n  Error: {e}")
        return
    except Exception as e:
        print(f"\n  Error during inference: {e}")
        return

    # ------------------------------------------------------------------
    # Step 4: Feature importance
    # ------------------------------------------------------------------
    sys.stdout.write("[4/4] Computing feature importance...  ")
    sys.stdout.flush()
    try:
        importance = compute_feature_importance(graph_data, model_path)
        print("Done")
    except Exception as e:
        print(f"\n  Warning: Could not compute feature importance: {e}")
        importance = {name: 0.0 for name in FEATURE_DISPLAY_NAMES}

    # ------------------------------------------------------------------
    # Fetch balance (non-critical, best effort)
    # ------------------------------------------------------------------
    balance = fetch_balance(address)

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------
    chart_paths: List[str] = []
    base_dir = args.output_dir
    # Create a subfolder per wallet: analyses/<address_short>/
    addr_short = address[:8] + "..." + address[-6:] if len(address) > 16 else address
    output_dir = os.path.join(base_dir, "analyses", addr_short)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_charts or args.open_charts:
        try:
            chart_paths.append(plot_feature_importance(importance, output_dir))
            chart_paths.append(plot_tx_volume(transactions, address, output_dir))
            chart_paths.append(plot_risk_gauge(inference["risk_score"],
                                               inference["classification"], output_dir))
        except Exception as e:
            print(f"\n  Warning: Chart generation failed: {e}")

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print_results(
        address=address,
        inference=inference,
        balance=balance,
        importance=importance,
        graph_data=graph_data,
        num_txs=num_txs,
        chart_paths=chart_paths,
    )

    # Open charts if requested
    if args.open_charts and chart_paths:
        import subprocess
        for path in chart_paths:
            if not path:
                continue
            try:
                if sys.platform == "darwin":
                    subprocess.Popen(["open", path])
                elif sys.platform.startswith("linux"):
                    subprocess.Popen(["xdg-open", path])
                elif sys.platform == "win32":
                    os.startfile(path)
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(
        description="Bitcoin Wallet Risk Analyzer — GNN-based classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python wallet_analyzer.py 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa\n"
            "  python wallet_analyzer.py --save-charts bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh\n"
            "  python wallet_analyzer.py --model my_model.pt 3FZbgi29cpjq2GjdwV8eyHuJJnkLtktZc5\n"
        ),
    )
    parser.add_argument("address", nargs="?", default=None,
                        help="Bitcoin wallet address to analyze")
    parser.add_argument("--model", default=None,
                        help="Path to GNN model file (.pt)")
    parser.add_argument("--output-dir", default=".",
                        help="Directory to save charts (default: current dir)")
    parser.add_argument("--save-charts", action="store_true",
                        help="Save charts (risk gauge, feature importance, tx volume)")
    parser.add_argument("--open-charts", action="store_true",
                        help="Save and open charts after analysis")

    args = parser.parse_args()

    print_header()

    # If address provided via command line, analyze it then enter interactive loop
    if args.address:
        analyze_address(args.address, args)
    else:
        # First prompt
        try:
            address = input("\n  Enter Bitcoin wallet address (or 'q' to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            sys.exit(0)

        if not address or address.lower() == 'q':
            print("\n  Goodbye!")
            sys.exit(0)

        analyze_address(address, args)

    # Interactive loop - keep asking for more addresses
    while True:
        try:
            print()
            address = input("  Enter another address (or 'q' to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye!")
            break

        if not address or address.lower() == 'q':
            print("\n  Goodbye!")
            break

        print_header()
        analyze_address(address, args)


if __name__ == "__main__":
    main()
