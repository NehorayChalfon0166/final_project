# Future Fix: Rebuild Training Data with Full Transaction Histories

## Problem

The training ego-graphs were built using only **25-50 transactions per wallet** because the mempool.space API returns max 25 txs per page and the graph pipeline's `api_fetcher.py` didn't paginate. This means:

- A wallet with 1,000 transactions only has its 25 most recent transactions in the training graph
- Center node features in training come from the **full** history (via REAL-CATS/Elliptic++ CSVs), but the ego-graph **edges and neighbors** only reflect recent activity
- At inference time, `fetch_transactions_mempool()` now fetches up to 500 txs (paginated), creating a train/inference mismatch in graph structure

## What Was Fixed (2026-04-13)

- `Backend/routes/utils/model_fit_utils.py` — `fetch_transactions_mempool()` now paginates up to 500 txs
- `standalone/wallet_analyzer.py` — `fetch_transactions()` now paginates up to 500 txs
- `src/graph/api_fetcher.py` — `fetch_address_transactions()` now paginates up to 500 txs

## What Still Needs Fixing

The existing cached transaction data and graphs in `graph_data/` were built with the old non-paginated fetcher (25-50 txs). To fully fix this:

### Steps

1. **Clear existing caches and graphs:**
   ```bash
   rm -rf graph_data/cache/train/*.json
   rm -rf graph_data/cache/test/*.json
   rm -rf graph_data/graphs/train/*.pt
   rm -rf graph_data/graphs/test/*.pt
   ```

2. **Reset progress files:**
   ```bash
   echo '{"completed":[],"failed":[],"total":0,"started_at":null,"last_updated":null}' > graph_data/metadata/progress_train.json
   echo '{"completed":[],"failed":[],"total":0,"started_at":null,"last_updated":null}' > graph_data/metadata/progress_test.json
   ```

3. **Rebuild all graphs** (will take much longer — ~20 API calls per wallet instead of 1):
   ```bash
   python run_pipeline.py --graphs --split both
   ```

4. **Retrain the model:**
   ```bash
   python run_pipeline.py --train --epochs 150
   ```

## Impact Assessment

- Current model works reasonably well on live wallets (10%-93% risk score range)
- Rebuilding would improve consistency between training and inference graph structures
- Time estimate: ~5x longer than original download due to pagination (at 0.5 req/s with ~20 pages per wallet = ~40s per wallet vs ~2s before)
- For 100k wallets: ~46 days at 0.5 req/s. Consider using multiple API endpoints or reducing the dataset size.

## Alternative: Limit Inference to Match Training

Instead of rebuilding training data, you could limit inference to 25 txs to match training:
```python
fetch_transactions_mempool(address, max_txs=25)
```
This is simpler but means the model won't use all available transaction data for new wallets.
