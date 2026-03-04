# Model Feature Mismatch - Fix Required

## Problem

The saved model (`outputs/gnn_model.pt`) was trained with **12 features**, but the current code expects **14 features**.

### Error Message
```
RuntimeError: Error(s) in loading state_dict for OptimalBitcoinGNN:
    size mismatch for conv1.lin_l.weight: copying a param with shape torch.Size([256, 12]) from checkpoint, the shape in current model is torch.Size([256, 14]).
    size mismatch for conv1.lin_r.weight: copying a param with shape torch.Size([256, 12]) from checkpoint, the shape in current model is torch.Size([256, 14]).
```

### Impact
- No confidence/prediction returned from `/api/v1/analyze/{address}`
- Feature importance endpoint returns 500 error
- UI cards for confidence, risk score, and feature importance don't display

---

## Root Cause

Two features were added to `src/graph/config.py` AFTER the model was trained:
- `min_sent_log` - Minimum sent amount (criminals avoid dust)
- `min_received_log` - Minimum received (benign have micro-txs)

---

## Solution Options

### Option A: Retrain the model (Recommended)
Retrain the model with the new 14 features to get improved accuracy.

### Option B: Revert to 12 features (Quick fix)

**File: `src/graph/config.py`**

Change from:
```python
FEATURE_COLUMNS = [
    'lifetime_seconds_log',
    'activity_rate_log',
    'in_out_balance_log',
    'total_txs_log',
    'send_receive_ratio_log',
    'fee_per_tx_log',
    'blocks_btwn_txs_mean_log',
    'fee_share_mean_log',
    'avg_tx_size_log',
    'tx_size_range_log',
    'max_sent_log',
    'max_received_log',
    'min_sent_log',        # REMOVE
    'min_received_log',    # REMOVE
]
NUM_NODE_FEATURES = len(FEATURE_COLUMNS)  # 14
```

To:
```python
FEATURE_COLUMNS = [
    'lifetime_seconds_log',
    'activity_rate_log',
    'in_out_balance_log',
    'total_txs_log',
    'send_receive_ratio_log',
    'fee_per_tx_log',
    'blocks_btwn_txs_mean_log',
    'fee_share_mean_log',
    'avg_tx_size_log',
    'tx_size_range_log',
    'max_sent_log',
    'max_received_log',
]
NUM_NODE_FEATURES = len(FEATURE_COLUMNS)  # 12
```

**File: `src/models/optimal_gnn.py`**

Change default from:
```python
num_node_features: int = 14,  # Updated: 12 original + 2 restored
```

To:
```python
num_node_features: int = 12,  # 12 features matching saved model
```

---

## Files Affected
- `src/graph/config.py` - Feature column definitions
- `src/models/optimal_gnn.py` - Model architecture defaults
- `Backend/routes/utils/model_fit_utils.py` - Feature names list (line 283-296)

---

## Date Identified
January 15, 2026
