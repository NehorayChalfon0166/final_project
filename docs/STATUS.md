# Project Status

Single source of truth for what's open, what's done, and the design context behind both. Replaces the split between `TODO_CHECKLIST.md` and the various `FUTURE_FIX_*.md` docs.

---

## Open

### Graph data download
Building ego-graphs for the full datasets is still in progress.
- Train: 9,499 / 86,876 wallets done (~11%)
- Test:  8,033 / 21,720 wallets done (~37%)

Resume with `python scripts/build_remaining_graphs.py` (use `--split train` or `--split test` to scope, `--max N` to test). The script reads progress from `graph_data/metadata/progress_*.json`; safe to interrupt and re-run.

### Rebuild training graphs at full pagination
Existing cached graphs in `graph_data/` were built before pagination was added — only 25 transactions per wallet were captured. Inference now paginates to 500 txs, creating a train/inference mismatch in graph *structure* (center-node features were always full-history; only the ego-graph edges were truncated).

Current model still works (10–93% risk range), so this is a quality improvement, not a correctness fix. Cost estimate: ~40s/wallet vs ~2s today, ~46 days for 100k wallets at 0.5 req/s — consider reducing dataset size or using multiple endpoints if pursuing.

Runbook:
```bash
rm -rf graph_data/cache/{train,test}/*.json graph_data/graphs/{train,test}/*.pt
echo '{"completed":[],"failed":[],"total":0,"started_at":null,"last_updated":null}' \
  | tee graph_data/metadata/progress_train.json graph_data/metadata/progress_test.json
python run_pipeline.py --graphs --split both
python run_pipeline.py --train --epochs 150
```

### Other improvements
- Train on the larger graph dataset once the download above completes.
- Add a validation split for temperature calibration (currently uses the test set, which biases the calibration metric).
- Re-run `python run_pipeline.py --evaluate` after any retraining to refresh evaluation artifacts.
- Consider focal loss for training — already implemented in `src/models/optimal_gnn.py`, currently not used.

---

## Recently Completed

Inference correctness
- Fix satoshi/BTC unit mismatch in `graph_builder.py` and `standalone/wallet_analyzer.py` (feature values were off by 30–90,000× vs training).
- Add feature clipping at inference (activity_rate, in_out_balance, send_receive_ratio, fee_share_mean) to match training pipeline.
- Fix `tx_size_range` to use sent-only amounts, matching training semantics.
- Switch mempool API from emzy.de (down) to mempool.space.

Inference performance
- Paginate transaction fetch to 500 txs in Backend, standalone CLI, and graph pipeline `api_fetcher` (was capped at 25).
- Process API requests sequentially in `api_fetcher` to avoid mempool.space rate limits.

Model architecture & calibration
- Add learnable ghost-node embeddings (replace zero-feature neighbors).
- Add node-type embedding (center vs ghost).
- Add hybrid readout (center + global mean pool + global max pool).
- Add DropEdge regularization.
- Add initial-connection residual through GNN layers.
- Remove label smoothing (it was compressing the probability range).
- Add post-hoc temperature scaling, applied at inference in both Backend and standalone.
- Retrain at the new architecture: 85.5% F1, well-calibrated probabilities.
- Revert from 14 features to 12 (7 correlated features removed). See `docs/archive/FUTURE_FIX_MODEL_MISMATCH.md` for the full mismatch story.

Tooling & UX
- Apply UI review accessibility & hierarchy fixes — verdict moved above model internals, button affordances added, contrast pass. Full review archived at `docs/archive/UI_REVIEW.md`.
- Cache wallet data and run analysis routes off the event loop in Backend.
- Anchor cumulative-balance chart to the API-reported current balance.
- Code review skill set up at `.claude/skills/code-review/SKILL.md`.

---

## Architecture & Calibration Notes

Why the model used to over-predict criminal: training fed BTC-scaled features through `log1p`, but inference passed raw satoshis. Five features (`fee_per_tx`, `avg_tx_size`, `max_sent`, `max_received`, `tx_size_range`) ended up 30–90,000× larger at inference than during training. The fix is in `src/graph/graph_builder.py` (and the standalone copy) — see commit history for details.

Why probabilities used to compress: zero-feature ghost neighbors created a constant bias through GATv2 attention, and `label_smoothing=0.1` trained the model toward [0.05, 0.95] instead of [0, 1]. Both were addressed by retraining with learnable ghost embeddings + a node-type embedding and dropping label smoothing. Post-hoc temperature scaling on top calibrates the remaining miscalibration.

Why the model file lives in `outputs/`: `outputs/gnn_model.pt` is the canonical weights file loaded by Backend, the standalone CLI, and `run_pipeline.py --evaluate`. `outputs/gnn_checkpoint.pt` is a fallback (full optimizer/scheduler state); `outputs/temperature.pt` is the calibration scaler. There used to be a `src/models/crypto_gnn_model.pt` from January — that file was stale and has been removed.
