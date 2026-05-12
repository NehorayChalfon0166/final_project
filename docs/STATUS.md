# Project Status

Single source of truth for what's open, what's done, and the design context behind both. Replaces the split between `TODO_CHECKLIST.md` and the various `FUTURE_FIX_*.md` docs.

For function-level call traces (offline pipeline, Backend `/analyze`, standalone CLI), see `docs/CALL_FLOW.md`.

---

## Directory layout

| Path | Role | Produced by | Consumed by |
|---|---|---|---|
| `data/` | Raw datasets (REAL-CATS, Elliptic++) | — | `--prepare` |
| `src/features/output/` | Engineered training CSVs | `--prepare` | `--graphs`, `--baseline` |
| `graph_data/cache/` | Cached mempool.space JSON per address | `--graphs` | `--graphs` (resume) |
| `graph_data/graphs/{train,test}/` | Materialized PyG `.pt` ego-graphs | `--graphs` | `--train`, `--evaluate` |
| `graph_data/metadata/` | Per-split build progress | `--graphs` | `--graphs` (resume) |
| `outputs/` | Trained model + eval/baseline artifacts | `--train`, `--baseline`, `--evaluate` | Backend, standalone |

The naming `src/features/output/` (training CSVs) vs `outputs/` (trained model + results) is unfortunate but stable — see the table for which is which.

---

## Open

### Graph data download
Building ego-graphs for the full datasets is still in progress.
- Train: 65,851 / 86,870 wallets done (~76%, 45 failed)
- Test:  14,737 / 21,718 wallets done (~68%, 18 failed)

Resume with `python run_pipeline.py --graphs --split train` (or `--split test`, or omit for both). The pipeline reads progress from `graph_data/metadata/progress_*.json`; safe to interrupt and re-run.

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

Refactor pass (cleanup branch)
- Removed legacy graph dirs `graph_data/{train,test}/` (canonical lives at `graph_data/graphs/{train,test}/`).
- Removed empty `src/evaluation/results/` and `src/baselines/results/` placeholders.
- Removed superseded scripts (`build_remaining_graphs.py`, `build_from_cache.py`, `check_model.py`, `evaluate_model.py`).
- Split `Backend/routes/utils/model_fit_utils.py` (416 lines) into a package: `_cache`, `_mempool`, `_graph`, `_model`, `_inference`. Public surface unchanged.
- Extracted `run_pipeline.py` step functions into `src/pipeline_steps/` (537 → 104 lines). CLI surface unchanged.
- Added `docs/CALL_FLOW.md` and `scripts/check_standalone_sync.py` for drift detection between standalone and `src/`.



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
