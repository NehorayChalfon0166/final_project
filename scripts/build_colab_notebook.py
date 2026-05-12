"""Build notebooks/colab_wallet_analyzer.ipynb from inline cell definitions.

Run once whenever the notebook source needs to change:
    python scripts/build_colab_notebook.py

The notebook is a fully self-contained Colab analyzer for the Bitcoin wallet
GNN classifier. Part 1 sets up deps + clones the public repo to load the model;
Part 2 takes a wallet address and produces verdict + visualizations.
"""
import json
import os


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


# =============================================================================
# PART 1 — SETUP
# =============================================================================

INTRO_MD = """# Bitcoin Wallet Risk Analyzer (Colab notebook)

Inference-only notebook for the GATv2-based wallet classifier
([repo](https://github.com/NehorayChalfon0166/final_project)). Given a Bitcoin
address, the notebook fetches its on-chain transactions from
[mempool.space](https://mempool.space), builds an ego-graph, runs it through the
trained model, and produces a verdict plus visualizations.

The notebook is split into **two parts**:

- **Part 1 — Setup**: run once per Colab session. Installs dependencies, clones
  the repo to get the model + the graph-builder source, defines helpers.
- **Part 2 — Analyze a wallet**: edit `WALLET_ADDRESS` and re-run as many times
  as you like. Each run produces:
  - verdict (CRIMINAL vs BENIGN), risk score, confidence
  - transactions-over-time bar chart (green = received, red = sent, weekly buckets) with totals beneath
  - cumulative balance over time (anchored to current balance)
  - feature importance (gradient saliency on the 12 input features)
  - top-5 neighbor classification (model run on each top counterparty)
  - ego-graph network viz with neighbors colored by GNN verdict
  - transaction table
"""

PART1_HEADER = """---
## Part 1 — Setup

Run all cells in this section once at the start of your Colab session. After
\"✓ Setup complete\" prints below, jump down to **Part 2**.
"""

INSTALL_CODE = """# Install dependencies. Colab usually has torch, numpy, pandas, matplotlib
# pre-installed; only torch-geometric and requests are routinely needed.
import subprocess, sys

def pip_install(*pkgs):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *pkgs])

pip_install('torch-geometric', 'requests')
print('✓ Dependencies installed')
"""

CLONE_CODE = """# Clone the public repo (shallow, ~few MB). Brings in:
#   - outputs/gnn_model.pt + outputs/temperature.pt  (trained weights)
#   - src/graph/graph_builder.py + src/graph/config.py
#   - src/models/optimal_gnn.py
# Pinned to the chore/cleanup-and-consolidate branch — it has the May-8
# retrained model. Switch to 'main' once PR #2 is merged.
import os, subprocess, sys

REPO_URL    = 'https://github.com/NehorayChalfon0166/final_project.git'
REPO_BRANCH = 'chore/cleanup-and-consolidate'
REPO_DIR    = '/content/final_project' if os.path.exists('/content') else os.path.abspath('final_project')

if not os.path.exists(REPO_DIR):
    subprocess.check_call(
        ['git', 'clone', '--depth', '1', '--branch', REPO_BRANCH, REPO_URL, REPO_DIR]
    )
else:
    print(f'Already cloned at {REPO_DIR}')

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
print(f'✓ Repo ready at {REPO_DIR}  (branch: {REPO_BRANCH})')
"""

IMPORTS_CODE = """import re
import time
from collections import defaultdict
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from IPython.display import HTML, display

from src.graph.config import (
    FEATURE_COLUMNS,
    NUM_EDGE_FEATURES,
    NUM_NODE_FEATURES,
)
from src.graph.graph_builder import EgoGraphBuilder
from src.models.optimal_gnn import OptimalBitcoinGNN

print('✓ Imports successful')
"""

LOAD_MODEL_CODE = """# Load the trained GATv2 model + temperature-scaling parameter.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = OptimalBitcoinGNN(
    num_node_features=NUM_NODE_FEATURES,
    num_edge_features=NUM_EDGE_FEATURES,
    hidden_dim=64,
).to(DEVICE)
model.load_state_dict(torch.load(
    f'{REPO_DIR}/outputs/gnn_model.pt',
    map_location=DEVICE,
    weights_only=True,
))
model.eval()

_temp_state = torch.load(f'{REPO_DIR}/outputs/temperature.pt', weights_only=True)
TEMPERATURE = float(_temp_state['temperature'])

n_params = sum(p.numel() for p in model.parameters())
print(f'✓ Model loaded ({n_params:,} parameters) on {DEVICE}')
print(f'✓ Temperature: {TEMPERATURE:.4f}')
"""

HELPERS_CODE = '''MEMPOOL_API = 'https://mempool.space/api'
SAT_TO_BTC = 1e-8

# --- address validation -----------------------------------------------------

def is_valid_bitcoin_address(addr: str) -> bool:
    """Accepts P2PKH/P2SH (1.../3...) and Bech32 (bc1...) formats."""
    return bool(
        re.match(r'^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$', addr)
        or re.match(r'^bc1[a-z0-9]{39,59}$', addr)
    )


# --- mempool.space HTTP -----------------------------------------------------

def _get(url: str, max_retries: int = 3, backoff: float = 5.0, timeout: int = 15):
    """GET with exponential-ish backoff on 429 / transient errors."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=timeout, headers=headers)
            if r.status_code == 429:
                time.sleep(backoff)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(backoff)
    raise RuntimeError(f'mempool.space request failed after {max_retries} tries: {last_err}')


def fetch_address_stats(addr: str) -> dict:
    """Returns chain_stats / mempool_stats / address."""
    return _get(f'{MEMPOOL_API}/address/{addr}')


def fetch_transactions(addr: str, max_pages: int = 2, sleep_between: float = 0.5) -> list:
    """Paginate /address/{addr}/txs. Cap = max_pages × ~50 confirmed txs."""
    first = _get(f'{MEMPOOL_API}/address/{addr}/txs')
    if not first:
        return []
    all_txs = list(first)
    last_id = first[-1]['txid']
    for _ in range(max_pages - 1):
        time.sleep(sleep_between)
        try:
            page = _get(f'{MEMPOOL_API}/address/{addr}/txs/chain/{last_id}')
        except RuntimeError:
            break
        if not page:
            break
        all_txs.extend(page)
        last_id = page[-1]['txid']
        if len(page) < 25:
            break
    return all_txs


# --- graph + model ----------------------------------------------------------

_builder = EgoGraphBuilder()


def build_graph(addr: str, txs: list):
    """Build ego-graph from raw mempool txs.

    Features are computed inline (compute_features_from_transactions) so the
    wallet does not need to exist in the training dataset.
    """
    return _builder.build_graph_for_new_address(addr, txs, label=-1)


def classify(graph) -> dict:
    """Run forward pass + temperature-scaled softmax. Returns verdict dict."""
    x = graph.x.to(DEVICE)
    edge_index = graph.edge_index.to(DEVICE)
    edge_attr = graph.edge_attr.to(DEVICE) if graph.edge_attr.numel() > 0 else None
    batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        logits = model(x, edge_index, edge_attr, batch)[0]
        probs = F.softmax(logits / TEMPERATURE, dim=0).cpu().numpy()
    prob_benign, prob_criminal = float(probs[0]), float(probs[1])
    return {
        'prob_benign': prob_benign,
        'prob_criminal': prob_criminal,
        'risk_score': prob_criminal,
        'classification': 'criminal' if prob_criminal > 0.5 else 'benign',
        'confidence': abs(prob_criminal - 0.5) * 2,
    }


def feature_importance(graph) -> dict:
    """Gradient saliency of the criminal-class logit w.r.t. center-node features.

    Absolute gradients, normalized to sum to 1.
    """
    x = graph.x.clone().detach().to(DEVICE).requires_grad_(True)
    edge_index = graph.edge_index.to(DEVICE)
    edge_attr = graph.edge_attr.to(DEVICE) if graph.edge_attr.numel() > 0 else None
    batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=DEVICE)
    out = model(x, edge_index, edge_attr, batch)
    out[0, 1].backward()
    grads = x.grad[0].abs().cpu().numpy()
    total = grads.sum()
    if total > 0:
        grads = grads / total
    return dict(zip(FEATURE_COLUMNS, grads.tolist()))


# --- neighbor extraction ----------------------------------------------------

def extract_neighbors_with_counts(addr: str, txs: list) -> list:
    """Return sorted neighbor list with tx counts + BTC flow.

    BTC attribution per shared tx:
    - target → neighbor (target appears in vin): output value going to neighbor
    - neighbor → target (target appears in vout): target's received value × the
      neighbor's share of total inputs. This is the share-based attribution
      block explorers use; without it a co-spender with a huge input would
      appear to have sent the target their full input regardless of where the
      money actually went.

    tx_count counts unique (neighbor, txid) pairs, so a neighbor that appears
    in two inputs of the same tx is one shared transaction, not two.
    """
    counts = defaultdict(int)
    btc_in = defaultdict(float)
    btc_out = defaultdict(float)
    for tx in txs:
        if not tx.get('status', {}).get('confirmed', False):
            continue
        is_sender = any(
            (inp.get('prevout') or {}).get('scriptpubkey_address') == addr
            for inp in tx.get('vin', [])
        )
        target_received = sum(
            out.get('value', 0)
            for out in tx.get('vout', [])
            if out.get('scriptpubkey_address') == addr
        )
        neighbors_in_tx = set()

        # target → neighbor: each output to a non-target address (only when
        # target is one of the senders).
        if is_sender:
            for out in tx.get('vout', []):
                a = out.get('scriptpubkey_address')
                if a and a != addr:
                    btc_out[a] += out.get('value', 0) * SAT_TO_BTC
                    neighbors_in_tx.add(a)

        # neighbor → target: target's received value, attributed proportionally
        # to each input contributor's share.
        if target_received > 0:
            total_input = sum(
                (inp.get('prevout') or {}).get('value', 0)
                for inp in tx.get('vin', [])
            )
            if total_input > 0:
                for inp in tx.get('vin', []):
                    prev = inp.get('prevout') or {}
                    a = prev.get('scriptpubkey_address')
                    if a and a != addr:
                        share = prev.get('value', 0) / total_input
                        btc_in[a] += target_received * share * SAT_TO_BTC
                        neighbors_in_tx.add(a)

        for n in neighbors_in_tx:
            counts[n] += 1

    rows = [
        {'address': a, 'tx_count': c, 'btc_in': btc_in[a], 'btc_out': btc_out[a]}
        for a, c in counts.items()
    ]
    rows.sort(key=lambda r: r['tx_count'], reverse=True)
    return rows


print('✓ Helpers defined')
'''

SETUP_DONE_MD = """✅ **Setup complete.** Continue to Part 2 below.
"""

# =============================================================================
# PART 2 — ANALYZE
# =============================================================================

PART2_HEADER = """---
## Part 2 — Analyze a wallet

Edit `WALLET_ADDRESS` below to the wallet you want to classify, then run all the
cells in this section in order. To analyze a different wallet, edit the address
and re-run Part 2 only (Part 1 state is preserved).
"""

WALLET_INPUT_CODE = """# Interactive prompt — Colab renders this as a text box.
# Press Enter without typing to use the default example wallet.
DEFAULT_WALLET = '17QAWGVpFV4gZ25NQug46e5mBho4uDP6MD'

_entered = input('Enter Bitcoin wallet address: ').strip()
WALLET_ADDRESS = _entered or DEFAULT_WALLET
print(f'Target: {WALLET_ADDRESS}' + ('  (default)' if not _entered else ''))
"""

FETCH_BUILD_CODE = """# Validate, fetch transactions, and build the ego-graph.
if not is_valid_bitcoin_address(WALLET_ADDRESS):
    raise ValueError(f'Invalid Bitcoin address format: {WALLET_ADDRESS!r}')

print(f'Analyzing {WALLET_ADDRESS} ...')
stats = fetch_address_stats(WALLET_ADDRESS)

chain = stats.get('chain_stats', {})
mp = stats.get('mempool_stats', {})
total_tx_count = chain.get('tx_count', 0) + mp.get('tx_count', 0)
print(f'  total on-chain txs reported by mempool.space: {total_tx_count:,}')

print('  fetching transactions (up to ~75: first page ~50 + 1 chain page ~25) ...')
txs = fetch_transactions(WALLET_ADDRESS, max_pages=2)
print(f'  fetched {len(txs)} transactions')

if not txs:
    raise RuntimeError('This wallet has no transactions; classification is not meaningful.')

graph = build_graph(WALLET_ADDRESS, txs)
print(f'\\n✓ Ego-graph built: {graph.num_nodes} nodes, {graph.num_edges} edges, '
      f'{graph.num_ghost_nodes} ghost neighbors')

if total_tx_count > len(txs):
    print(f'⚠ Partial graph: fetched {len(txs)} of {total_tx_count:,} txs '
          f'(notebook caps at 2 pages to keep runtime short).')
"""

CLASSIFY_CODE = '''# Run the model and display the verdict.
result = classify(graph)

is_crim = result['classification'] == 'criminal'
color = '#D84315' if is_crim else '#2E7D32'
bg = '#FFEBEE' if is_crim else '#E8F5E9'
verdict = result['classification'].upper()

display(HTML(f"""
<div style="border-left: 6px solid {color}; background: {bg}; color: #1A1A1A; padding: 16px 18px; margin: 10px 0; border-radius: 6px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
  <div style="font-size: 22px; color: {color}; font-weight: 700; margin-bottom: 10px;">{verdict}</div>
  <div style="font-size: 14px; line-height: 1.6; color: #1A1A1A;">
    Risk score: <b style="color: #000;">{result['risk_score']:.4f}</b><br>
    Confidence: <b style="color: #000;">{result['confidence']:.4f}</b>
  </div>
</div>
"""))
'''

TX_TIMELINE_CODE = """# Per-tx records (reused by the transaction-table cell).
from datetime import timedelta as _td

tx_records = []
for _tx in txs:
    if not _tx.get('status', {}).get('confirmed', False):
        continue
    _bt = _tx['status'].get('block_time', 0)
    if _bt == 0:
        continue
    _is_sender = any(
        (inp.get('prevout') or {}).get('scriptpubkey_address') == WALLET_ADDRESS
        for inp in _tx.get('vin', [])
    )
    _received = sum(
        out.get('value', 0)
        for out in _tx.get('vout', [])
        if out.get('scriptpubkey_address') == WALLET_ADDRESS
    )
    _sent_to_others = sum(
        out.get('value', 0)
        for out in _tx.get('vout', [])
        if out.get('scriptpubkey_address') != WALLET_ADDRESS
    ) if _is_sender else 0
    if _is_sender and _received == 0:
        _direction = 'out'; _amount_sats = _sent_to_others
    elif _is_sender and _received > 0:
        _direction = 'in';  _amount_sats = _received    # self-transfer / change
    else:
        _direction = 'in';  _amount_sats = _received
    tx_records.append({
        'txid': _tx['txid'],
        'datetime': datetime.fromtimestamp(_bt, tz=timezone.utc),
        'block_time': _bt,
        'block_height': _tx['status'].get('block_height'),
        'direction': _direction,
        'amount_btc': _amount_sats * SAT_TO_BTC,
        'amount_sats': _amount_sats,
        'fee_sats': _tx.get('fee', 0),
        'is_sender': _is_sender,
        'received_sats': _received,
        'sent_sats': _sent_to_others,
    })
tx_records.sort(key=lambda r: r['block_time'])

# Aggregate into weekly buckets: BTC received vs sent.
_weekly = {}
for _r in tx_records:
    _wk = (_r['datetime'] - _td(days=_r['datetime'].weekday())).strftime('%Y-%m-%d')
    _b = _weekly.setdefault(_wk, {'in': 0.0, 'out': 0.0})
    if _r['received_sats'] > 0:
        _b['in']  += _r['received_sats']  * SAT_TO_BTC
    if _r['is_sender'] and _r['sent_sats'] > 0:
        _b['out'] += _r['sent_sats'] * SAT_TO_BTC

CLR_GREEN, CLR_RED = '#22c55e', '#ef4444'
CLR_GRAY500, CLR_GRAY700, CLR_GRAY200 = '#6b7280', '#374151', '#e5e7eb'

if _weekly:
    _weeks = sorted(_weekly.keys())
    _inc = [_weekly[w]['in']  for w in _weeks]
    _outv = [_weekly[w]['out'] for w in _weeks]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('white')
    _x = np.arange(len(_weeks)); _w = 0.4
    ax.bar(_x - _w/2, _inc,  _w, label='Received', color=CLR_GREEN, edgecolor='none')
    ax.bar(_x + _w/2, _outv, _w, label='Sent',     color=CLR_RED,   edgecolor='none')
    ax.set_ylabel('BTC', fontsize=12, color=CLR_GRAY500)
    ax.set_title('Transaction Volume (Weekly)', fontsize=16, fontweight='bold',
                 color=CLR_GRAY700, pad=15)
    if len(_weeks) > 20:
        _step = max(1, len(_weeks) // 10)
        _ticks = list(range(0, len(_weeks), _step))
        ax.set_xticks([_x[i] for i in _ticks])
        ax.set_xticklabels([_weeks[i] for i in _ticks], rotation=45, ha='right',
                           fontsize=9, color=CLR_GRAY500)
    else:
        ax.set_xticks(_x)
        ax.set_xticklabels(_weeks, rotation=45, ha='right', fontsize=9, color=CLR_GRAY500)
    ax.tick_params(axis='y', colors=CLR_GRAY500, labelsize=10)
    ax.grid(axis='y', ls='--', alpha=0.3, color=CLR_GRAY200); ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(CLR_GRAY200); ax.spines['bottom'].set_color(CLR_GRAY200)
    ax.legend(frameon=False, loc='upper left')
    plt.tight_layout(); plt.show()
else:
    print('No confirmed transactions to chart.')

# Totals below the chart (replaces the old separate BTC-overview chart).
_chain = stats.get('chain_stats', {})
_mp    = stats.get('mempool_stats', {})
_funded = (_chain.get('funded_txo_sum', 0) + _mp.get('funded_txo_sum', 0)) * SAT_TO_BTC
_spent  = (_chain.get('spent_txo_sum',  0) + _mp.get('spent_txo_sum',  0)) * SAT_TO_BTC
_bal    = _funded - _spent
_nfunded = _chain.get('funded_txo_count', 0) + _mp.get('funded_txo_count', 0)
_nspent  = _chain.get('spent_txo_count',  0) + _mp.get('spent_txo_count',  0)

display(HTML(f'''
<div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; color:#374151;
            background:#f9fafb; border:1px solid #e5e7eb; border-radius:8px;
            padding:14px 18px; margin-top:6px; max-width:720px;">
  <table style="border-collapse:collapse; width:100%; font-size:14px;">
    <tr><td style="padding:4px 0; color:#6b7280;">Total received</td>
        <td style="padding:4px 0; text-align:right; color:#22c55e; font-weight:600;">
            {_funded:.8f} BTC  ·  {_nfunded:,} txo</td></tr>
    <tr><td style="padding:4px 0; color:#6b7280;">Total sent</td>
        <td style="padding:4px 0; text-align:right; color:#ef4444; font-weight:600;">
            {_spent:.8f} BTC  ·  {_nspent:,} txo</td></tr>
    <tr><td style="padding:4px 0; color:#6b7280;">Current balance</td>
        <td style="padding:4px 0; text-align:right; color:#1f2937; font-weight:700;">
            {_bal:.8f} BTC</td></tr>
    <tr><td style="padding:4px 0; color:#6b7280;">Total transactions</td>
        <td style="padding:4px 0; text-align:right; color:#1f2937;">
            {total_tx_count:,}</td></tr>
  </table>
</div>
'''))
"""

CUMULATIVE_BALANCE_CODE = """# Cumulative balance over time, anchored to the current balance reported by
# mempool.space. We walk events backwards so the curve always ends at the
# real current balance, even when mempool capped older confirmed txs.
_events = []  # (timestamp, delta_btc)
for _r in tx_records:
    _delta = (_r['received_sats'] - _r['sent_sats']) * SAT_TO_BTC
    _events.append((_r['block_time'], _delta))
_events.sort(key=lambda e: e[0])

if _events:
    _times_dt = [datetime.fromtimestamp(e[0], tz=timezone.utc) for e in _events]
    _deltas = [e[1] for e in _events]
    _end_balance = _bal  # from the totals block above
    _start_balance = _end_balance - sum(_deltas)
    _cum = _start_balance + np.cumsum(_deltas)

    CLR_BLUE = '#3b82f6'
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('white')
    ax.fill_between(_times_dt, _cum, alpha=0.25, color=CLR_BLUE)
    ax.plot(_times_dt, _cum, color=CLR_BLUE, lw=2)
    ax.set_ylabel('Balance (BTC)', fontsize=12, color=CLR_GRAY500)
    ax.set_title('Cumulative Balance Over Time', fontsize=16, fontweight='bold',
                 color=CLR_GRAY700, pad=15)
    ax.tick_params(axis='both', colors=CLR_GRAY500, labelsize=10)
    ax.grid(axis='y', ls='--', alpha=0.3, color=CLR_GRAY200); ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(CLR_GRAY200); ax.spines['bottom'].set_color(CLR_GRAY200)
    fig.autofmt_xdate()
    ax.text(0.5, -0.18,
            f'Current balance: {_end_balance:.8f} BTC   '
            f'(showing {len(_events)} of {total_tx_count:,} txs)',
            transform=ax.transAxes, ha='center', fontsize=10, color=CLR_GRAY500)
    plt.tight_layout(); plt.show()
else:
    print('No confirmed transactions — cannot reconstruct balance history.')
"""

EGOGRAPH_VIZ_CODE = """# Ego-graph network visualization. Reuses the verdicts + BTC flows already
# computed in the neighbor-classification cell (neighbors_df).
import networkx as nx

_verdict_map, _flow_map = {}, {}
try:
    for _row in neighbors_df.to_dict('records'):
        _verdict_map[_row['neighbor']] = _row['verdict']
        _flow_map[_row['neighbor']] = (_row['btc_in'], _row['btc_out'])
except NameError:
    # Neighbor cell wasn't run — fall back to gray nodes with flows recomputed.
    for _n in extract_neighbors_with_counts(WALLET_ADDRESS, txs)[:5]:
        _verdict_map[_n['address']] = 'unknown'
        _flow_map[_n['address']] = (_n['btc_in'], _n['btc_out'])

_top5_addrs = list(_verdict_map.keys())

# Build directed graph. Edge direction = dominant flow; label = net BTC.
G = nx.DiGraph()
G.add_node(WALLET_ADDRESS, kind='center')
for _a in _top5_addrs:
    G.add_node(_a, kind='neighbor', verdict=_verdict_map.get(_a, 'unknown'))
    _bin, _bout = _flow_map.get(_a, (0.0, 0.0))
    _net = _bin - _bout   # +ve: net inflow to center; -ve: net outflow
    if _net >= 0:
        G.add_edge(_a, WALLET_ADDRESS, btc=_bin, label=f'{_bin:.6f} BTC →')
    else:
        G.add_edge(WALLET_ADDRESS, _a, btc=_bout, label=f'→ {_bout:.6f} BTC')

# Layout: center fixed, neighbors on a circle
pos = nx.spring_layout(G, k=1.8, iterations=60, seed=42)
pos[WALLET_ADDRESS] = np.array([0.0, 0.0])

# Colors
_center_color = '#ef4444' if result['classification'] == 'criminal' else '#22c55e'
def _v_color(v):
    if v == 'CRIMINAL': return '#ef4444'
    if v == 'BENIGN':   return '#22c55e'
    return '#9ca3af'  # unknown / no data / error

_node_colors = [
    _center_color if n == WALLET_ADDRESS else _v_color(G.nodes[n].get('verdict', 'unknown'))
    for n in G.nodes()
]
_node_sizes = [2400 if n == WALLET_ADDRESS else 1100 for n in G.nodes()]

fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor('white')
nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=18,
                       edge_color='#9ca3af', width=1.6, alpha=0.7,
                       connectionstyle='arc3,rad=0.05')
nx.draw_networkx_nodes(G, pos, ax=ax, node_color=_node_colors,
                       node_size=_node_sizes, edgecolors='#374151', linewidths=1.5)
# Labels: short address (first 8 chars) + verdict for neighbors. matplotlib
# treats these as plain text, but address chars are base58/bech32-safe anyway.
_labels = {WALLET_ADDRESS: f'{WALLET_ADDRESS[:8]}…\\nTARGET'}
for _a in _top5_addrs:
    _labels[_a] = f'{_a[:8]}…\\n{G.nodes[_a].get("verdict", "unknown")}'
nx.draw_networkx_labels(G, pos, labels=_labels, font_size=9,
                        font_color='#111827', ax=ax)

# Edge labels: BTC flow (precomputed above)
_edge_labels = {(u, v): d.get('label', f'{d["btc"]:.6f} BTC')
                for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=_edge_labels,
                             font_size=8, font_color='#4b5563',
                             bbox=dict(boxstyle='round,pad=0.2',
                                       fc='white', ec='none', alpha=0.9),
                             ax=ax)

ax.set_title('Ego-graph: target wallet + top-5 neighbors  ·  colored by GNN verdict',
             fontsize=14, fontweight='bold', color=CLR_GRAY700, pad=10)
ax.axis('off')
# Legend
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor='#22c55e', edgecolor='#374151', label='BENIGN'),
    Patch(facecolor='#ef4444', edgecolor='#374151', label='CRIMINAL'),
    Patch(facecolor='#9ca3af', edgecolor='#374151', label='unknown / no data'),
], loc='upper right', frameon=True, fontsize=10)
plt.tight_layout(); plt.show()
"""

FEATURE_IMPORTANCE_CODE = """# Gradient saliency of the criminal-class logit w.r.t. center-node features.
fi = feature_importance(graph)
fi_sorted_asc = sorted(fi.items(), key=lambda kv: kv[1])
_names = [k for k, _ in fi_sorted_asc]
_vals  = [v for _, v in fi_sorted_asc]

_top3 = {k for k, _ in sorted(fi.items(), key=lambda kv: -kv[1])[:3]}
_colors = ['#FF6F00' if n in _top3 else '#42A5F5' for n in _names]

fig, ax = plt.subplots(figsize=(9.5, 5))
ax.barh(_names, _vals, color=_colors, edgecolor='#37474F', linewidth=0.4)
ax.set_xlabel('normalized |gradient|')
ax.set_title('Feature importance (top 3 highlighted)')
ax.grid(True, axis='x', alpha=0.3)
plt.tight_layout(); plt.show()

print('Top 3:')
for _i, (_n, _v) in enumerate(sorted(fi.items(), key=lambda kv: -kv[1])[:3], 1):
    print(f'  {_i}. {_n}: {_v:.4f}')
"""

NEIGHBOR_CODE = """# Classify the top-5 most-frequent counterparty wallets.
# Each neighbor needs its own mempool fetch (~2 s due to rate-limit). Expect ~10 s total.
print('Identifying top neighbors by tx count ...')
_neighbors = extract_neighbors_with_counts(WALLET_ADDRESS, txs)[:5]
print(f'  found {len(_neighbors)} top neighbors\\n')

_rows = []
for _i, _n in enumerate(_neighbors, 1):
    _addr = _n['address']
    print(f'  [{_i}/{len(_neighbors)}] fetching {_addr[:16]}... ', end='', flush=True)
    try:
        _n_txs = fetch_transactions(_addr, max_pages=2)
        if not _n_txs:
            _verdict = 'no data'
            _risk = None
        else:
            _n_graph = build_graph(_addr, _n_txs)
            _r = classify(_n_graph)
            _verdict = _r['classification'].upper()
            _risk = round(_r['risk_score'], 4)
        print(_verdict)
    except Exception as _e:
        _verdict = 'error'
        _risk = None
        print(f'failed ({_e})')
    _rows.append({
        'neighbor': _addr,
        'tx_count_with_target': _n['tx_count'],
        'btc_in':  round(_n['btc_in'], 6),
        'btc_out': round(_n['btc_out'], 6),
        'verdict': _verdict,
        'risk_score': _risk,
    })

neighbors_df = pd.DataFrame(_rows)
print()
display(neighbors_df)
"""

TX_TABLE_CODE = '''# Transaction history as a styled HTML table with clickable txid links.
def _fmt_amount(sats):
    if abs(sats) >= 1_000_000:
        return f'{sats / 1e8:.8f} BTC'
    return f'{sats:,} sats'


def _truncate_txid(s, head=12, tail=6):
    return s[:head] + '...' + s[-tail:] if len(s) > head + tail else s


_recent = sorted(tx_records, key=lambda r: r['block_time'], reverse=True)[:50]

_row_html = []
for _r in _recent:
    _dir = _r['direction']
    if _dir == 'in':
        _dir_html = '<span style="color:#22c55e; font-weight:600">RECV</span>'
    else:
        _dir_html = '<span style="color:#ef4444; font-weight:600">SENT</span>'

    _link = (f'<a href="https://mempool.space/tx/{_r["txid"]}" target="_blank" '
             f'style="color:#3b82f6; text-decoration:none">'
             f'{_truncate_txid(_r["txid"])}</a>')
    _block = _r['block_height'] if _r['block_height'] is not None else ''
    _row_html.append(
        f'<tr><td>{_link}</td><td>{_dir_html}</td>'
        f'<td>{_fmt_amount(_r["amount_sats"])}</td>'
        f'<td>{_fmt_amount(_r["fee_sats"])}</td>'
        f'<td>{_block}</td>'
        f'<td>{_r["datetime"].strftime("%Y-%m-%d %H:%M")}</td></tr>'
    )

_table_html = f"""
<style>
.tx-table {{ border-collapse: collapse; width: 100%; font-family: monospace; font-size: 12px; }}
.tx-table th {{ background: #f3f4f6; color: #374151; padding: 8px 10px; text-align: left;
               border-bottom: 2px solid #e5e7eb; }}
.tx-table td {{ padding: 6px 10px; border-bottom: 1px solid #f3f4f6; color: #4b5563; }}
.tx-table tr:hover {{ background: #f9fafb; }}
.tx-table a:hover {{ text-decoration: underline !important; }}
</style>
<h3 style="color:#374151; font-family:sans-serif; margin-bottom:8px;">Transaction History
  <span style="font-weight:normal; font-size:14px; color:#6b7280;">({len(tx_records)} confirmed, showing up to 50)</span>
</h3>
<table class="tx-table">
<tr><th>TXID</th><th>Direction</th><th>Amount</th><th>Fee</th><th>Block</th><th>Date (UTC)</th></tr>
{"".join(_row_html)}
</table>
"""
display(HTML(_table_html))
'''


def main():
    cells = [
        md(INTRO_MD),
        md(PART1_HEADER),
        code(INSTALL_CODE),
        code(CLONE_CODE),
        code(IMPORTS_CODE),
        code(LOAD_MODEL_CODE),
        code(HELPERS_CODE),
        md(SETUP_DONE_MD),
        md(PART2_HEADER),
        code(WALLET_INPUT_CODE),
        code(FETCH_BUILD_CODE),
        code(CLASSIFY_CODE),
        md("### Transactions over time"),
        code(TX_TIMELINE_CODE),
        md("### Cumulative balance over time"),
        code(CUMULATIVE_BALANCE_CODE),
        md("### Feature importance"),
        code(FEATURE_IMPORTANCE_CODE),
        md("### Top-5 neighbor classification"),
        code(NEIGHBOR_CODE),
        md("### Ego-graph network visualization"),
        code(EGOGRAPH_VIZ_CODE),
        md("### Transaction table"),
        code(TX_TABLE_CODE),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(project_root, 'notebooks', 'colab_wallet_analyzer.ipynb')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    print(f'Wrote {out_path}')
    print(f'  cells: {len(cells)} ({sum(1 for c in cells if c["cell_type"] == "code")} code, '
          f'{sum(1 for c in cells if c["cell_type"] == "markdown")} markdown)')


if __name__ == '__main__':
    main()
