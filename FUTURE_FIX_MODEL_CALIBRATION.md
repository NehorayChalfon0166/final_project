# Fix GNN Model Prediction Calibration

## Context

The Bitcoin wallet GNN classifier never outputs low criminal probabilities — it produces either ~98% or ~40-46%, never below ~40%. This makes the model useless for identifying clearly benign wallets. Investigation found **3 root causes**, with the first being a clear bug.

## Root Causes (priority order)

### 1. Satoshi/BTC Unit Mismatch (BUG — primary cause)

**Training**: `pipeline_conservative.py` calls `normalize_btc_values()` which detects satoshi-denominated values (median > 1000) and divides by 100M to convert to BTC *before* log1p.

**Inference**: `graph_builder.py:compute_features_from_transactions()` uses raw mempool API values which are in **satoshis** and never converts to BTC.

Result: 5 features are off by ~30-90,000x after log1p:
| Feature | Training (BTC) | Inference (sats) | Scale diff |
|---|---|---|---|
| fee_per_tx | log1p(0.0001) = 0.0001 | log1p(10000) = 9.21 | 92,000x |
| avg_tx_size | log1p(0.75) = 0.56 | log1p(75M) = 18.13 | 32x |
| max_sent | log1p(0.5) = 0.41 | log1p(50M) = 17.73 | 44x |
| max_received | log1p(1.0) = 0.69 | log1p(100M) = 18.42 | 27x |
| tx_size_range | log1p(0.5) = 0.41 | log1p(50M) = 17.73 | 44x |

### 2. Missing Feature Clipping (minor)

Training clips `activity_rate` to [0, 1000], `in_out_balance` to [0, 100], `send_receive_ratio` to [0, 100]. Inference doesn't clip at all.

### 3. tx_size_range Semantic Mismatch (minor)

Training: `max_sent - min_sent` (sent only). Inference: `max(all_amounts) - min(all_amounts)` (sent + received combined).

### 4. Ghost Node Zero Features (model-level, deferred)

All neighbor nodes have zero features. The GATv2 attention aggregation over these zero-vectors creates a constant bias that compresses probability outputs. This is a training-time issue that requires retraining with learnable ghost embeddings — defer to a separate task.

### 5. Label Smoothing Compression (model-level, deferred)

`label_smoothing=0.1` trains the model to target [0.05, 0.95] instead of [0, 1], compressing the output range. Combined with ghost-node bias, this contributes to the floor. Fix requires retraining — defer.

## Plan

### Step 1: Fix satoshi→BTC conversion in `compute_features_from_transactions()`

**File**: `src/graph/graph_builder.py` (lines 276-390)

Convert monetary values from satoshis to BTC before computing derived features:
- Add `SAT_TO_BTC = 1e-8` constant
- After the transaction loop, convert: `total_sent`, `total_received`, `total_fees`, all entries in `sent_amounts`, all entries in `received_amounts`
- This ensures `fee_per_tx`, `avg_tx_size`, `tx_size_range`, `max_sent`, `max_received`, `send_receive_ratio`, `fee_share_mean` all use BTC-scale values

### Step 2: Add feature clipping to match training pipeline

**File**: `src/graph/graph_builder.py`

After computing derived features, clip to match training:
- `activity_rate = clip(activity_rate, 0, 1000)`
- `in_out_balance = clip(in_out_balance, 0, 100)` 
- `send_receive_ratio = clip(send_receive_ratio, 0, 100)`
- `fee_share_mean = clip(fee_share_mean, 0, 1)`

### Step 3: Fix tx_size_range to match training semantics

**File**: `src/graph/graph_builder.py`

Change from `max(all_amounts) - min(all_amounts)` to `max(sent_amounts) - min(sent_amounts)` to match training.

### Step 4: Also fix the standalone analyzer

**File**: `standalone/wallet_analyzer.py`

Check if it has the same `compute_features_from_transactions()` — if so, apply the same fixes.

## Files to Modify

- `src/graph/graph_builder.py` — Main fix (steps 1-3)
- `standalone/wallet_analyzer.py` — Same fix if it has its own copy

## Verification

1. Run the backend and analyze a known benign wallet (e.g., Satoshi's address `1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa`) — should show low risk score
2. Analyze a few random wallets — should see varied probabilities across the full range, not just 40-46% or 98%
3. Print feature values before/after the fix for a test wallet to confirm they're now in the same scale as training data
4. Run the standalone analyzer on the same wallet to confirm consistent results
