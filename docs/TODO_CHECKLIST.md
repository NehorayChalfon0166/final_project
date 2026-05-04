# Project Checklist

## Completed

- [x] Fix satoshi/BTC unit mismatch in inference features (`graph_builder.py`, `wallet_analyzer.py`)
- [x] Add feature clipping to match training pipeline (activity_rate, in_out_balance, send_receive_ratio, fee_share_mean)
- [x] Fix tx_size_range to use sent-only amounts matching training semantics
- [x] Add learnable ghost node embeddings to replace zero-feature neighbors
- [x] Add node-type embedding (center vs ghost distinction)
- [x] Add hybrid readout (center node + global mean pool + global max pool)
- [x] Add DropEdge regularization
- [x] Add initial connection residual (preserves original signal through GNN layers)
- [x] Remove label smoothing (was compressing probability range)
- [x] Add post-hoc temperature scaling calibration
- [x] Apply temperature scaling at inference (backend + standalone)
- [x] Switch mempool API from emzy.de (down) to mempool.space
- [x] Fix api_fetcher to process requests sequentially (avoid rate limits)
- [x] Add transaction pagination to inference (fetch up to 500 txs instead of 25)
- [x] Add transaction pagination to graph pipeline api_fetcher
- [x] Add transaction pagination to standalone wallet_analyzer
- [x] Retrain model with improved architecture (85.5% F1, well-calibrated probabilities)
- [x] Update standalone wallet_analyzer with new model architecture
- [x] Set up code review skill (`.claude/skills/code-review/SKILL.md`)
- [x] Build graphs from all cached transactions
- [x] Sync progress files with actual graph counts

## In Progress

- [ ] Download remaining wallet graph data (train: 9,499/86,876 done, test: 8,033/21,720 done)

## Future Improvements

- [ ] Rebuild training graphs with full transaction pagination (see `FUTURE_FIX_TRANSACTION_PAGINATION.md`)
- [ ] Retrain model on larger dataset once more graphs are downloaded
- [ ] Add validation split for more robust temperature calibration (currently uses test set)
- [ ] Consider focal loss for training (already implemented, not currently used)
- [ ] Run full evaluation pipeline (`python run_pipeline.py --evaluate`) after retraining
