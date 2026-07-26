# Call Flow

Function-level view of how the two entry points (training pipeline, FastAPI backend) move through the codebase. Both load the same trained model from `outputs/gnn_model.pt` + `outputs/temperature.pt`.

---

## Repo layout, by role in the flow

```
data/                   raw datasets (REAL-CATS, Elliptic++)              [INPUT to --prepare]
src/features/output/    engineered training CSVs                          [PRODUCED by --prepare, CONSUMED by --graphs/--baseline]
graph_data/cache/       per-address cached mempool.space JSON             [PRODUCED+CONSUMED by --graphs]
graph_data/graphs/      materialized PyG .pt ego-graphs                   [PRODUCED by --graphs, CONSUMED by --train/--evaluate]
graph_data/metadata/    per-split build progress                          [PRODUCED+CONSUMED by --graphs]
outputs/                trained model + eval/baseline artifacts           [PRODUCED by --train/--baseline/--evaluate, CONSUMED by Backend]
Backend/                FastAPI app                                       [serves Frontend, reads outputs/]
Frontend/               React+Vite SPA                                    [calls Backend]
```

---

## Offline pipeline: `python run_pipeline.py --prepare --graphs --baseline --train --evaluate`

```
run_pipeline.main()
├─ step_prepare_data()
│  ├─ src/features/pipeline_conservative.py::main()
│  │   ├─ load_realcats() / load_elliptic()
│  │   ├─ extract_shared_features → normalize_btc_values
│  │   ├─ calculate_additional_features → engineer_features
│  │   └─ add_log_transforms
│  │       → src/features/output/conservative_feature_matrix_with_logs.csv
│  └─ src/features/prepare_balanced_dataset.py::main(use_log=True)
│      ├─ load_feature_matrix → analyze_label_distribution
│      ├─ create_balanced_dataset (CUTOFF=2500 per (label, source))
│      ├─ select_features → create_train_test_split (stratified 80/20)
│      └─ save_datasets
│          → src/features/output/{train,test,balanced_training}_dataset.csv
│          → src/features/output/feature_columns.txt
│
├─ step_build_graphs(split='both')
│  └─ for split in ['train', 'test']:
│     src/graph/pipeline.py::GraphConstructionPipeline(split).run()
│       ├─ get_pending_addresses()         (CSV ↔ existing .pt diff)
│       └─ asyncio.run(process_batch())
│           ├─ src/graph/api_fetcher.py::MempoolFetcher.fetch_address_transactions()
│           │   └─ cache_manager.get_cached_transactions / save_transactions
│           ├─ src/graph/graph_builder.py::EgoGraphBuilder.build_ego_graph()
│           │   └─ _parse_transactions → torch_geometric.data.Data
│           └─ cache_manager.save_graph
│               → graph_data/graphs/{split}/{addr}.pt
│
├─ step_train_baseline()
│  └─ src/baselines/xgboost_baseline.py::XGBoostBaseline
│     ├─ load_data (FEATURE_COLUMNS from src/graph/config.py)
│     ├─ train (5-fold stratified CV + final fit on all)
│     └─ outputs/baselines/{xgboost_results.json, xgboost_model.pkl}
│
├─ step_train_gnn()
│  ├─ src/graph/dataloader.py::get_train_val_loaders / get_test_loader
│  │   └─ EgoGraphDataset reads graph_data/graphs/{split}/*.pt
│  ├─ src/models/optimal_gnn.py::OptimalBitcoinGNN
│  │     (3× GATv2Conv multi-head, residual, hybrid readout center+mean+max)
│  ├─ src/models/train_optimal.py::train_epoch / evaluate (per-epoch loop)
│  ├─ TemperatureScaler.calibrate
│  │   → outputs/temperature.pt
│  └─ outputs/{gnn_model.pt, gnn_checkpoint.pt, gnn_training_history.json}
│
└─ step_evaluate()
   └─ src/evaluation/run_evaluation.py::run_standard_evaluation()
      ├─ load OptimalBitcoinGNN(outputs/gnn_model.pt) + temperature
      ├─ src/evaluation/metrics.py::compute_metrics
      │   → outputs/evaluation/evaluation_results.json
      ├─ src/evaluation/error_analysis.py::ErrorAnalyzer
      │   → outputs/evaluation/gnn_false_{positives,negatives}.csv
      └─ src/evaluation/interpretability.py::ModelInterpreter
          → outputs/evaluation/feature_importance.{json,png}
```

The `--evaluate` step runs purely on cached `.pt` graphs — no API calls.

---

## Online: `POST /api/v1/analyze/{address}` (FastAPI Backend)

```
Backend/main.py (startup)
  └─ model_fit_utils.get_cached_model() preloads OptimalBitcoinGNN + temperature

Backend/routes/wallet.py::analyze_wallet(address)
  ├─ is_valid_bitcoin_address(address)
  └─ Backend/routes/utils/model_fit_utils.py::analyze_wallet_pipeline()
     ├─ get_cached_transactions(addr)        (TTL 300s, in-flight dedup)
     │   └─ fetch_transactions_mempool()
     │       → mempool.space/api/address/{addr}/txs/chain
     ├─ get_cached_graph(addr, txs)           (TTL 300s)
     │   └─ src/graph/graph_builder.py::EgoGraphBuilder.build_graph_for_new_address()
     │       → torch_geometric.data.Data (12 node features, 3 edge features)
     └─ get_cached_model() → OptimalBitcoinGNN.forward()
         logits / temperature → softmax → {prob_benign, prob_criminal,
                                           risk_score, classification}
```

Other routes (all under `/api/v1`):

- `GET /info/{address}` — raw mempool.space stats (balance, totals, tx count). Cached.
- `GET /feature-importance/{address}` — gradient saliency on the cached graph; reuses `_GRAPH_CACHE`.
- `GET /random` — picks an address from `data/realcats` for the demo button.
- `GET /health`, `GET /ping` — Backend/routes/health.py.

In-memory caches (Backend/routes/utils/model_fit_utils.py):

- `_MODEL_CACHE` — model + temperature + device, persistent across requests.
- `_TX_CACHE`, `_STATS_CACHE`, `_GRAPH_CACHE` — per-address, TTL 300s.
- `_TX_INFLIGHT`, `_STATS_INFLIGHT` — threading events to dedup concurrent fetches for the same address.

---

## When to look at which file

- "Why does the model say X about wallet Y?"
  → start at `Backend/routes/utils/model_fit_utils.py::analyze_wallet_pipeline`, walk into `src/graph/graph_builder.py` (features), then `src/models/optimal_gnn.py` (forward).
- "How were the training graphs built?"
  → `src/graph/pipeline.py` → `src/graph/api_fetcher.py` + `src/graph/graph_builder.py`.
- "What did training optimize?"
  → `src/models/train_optimal.py` (loop, criterion, scheduler, calibration).
- "How is performance measured?"
  → `src/evaluation/run_evaluation.py` → `metrics.py`, `error_analysis.py`, `interpretability.py`.
- "What does the Frontend show?"
  → `Frontend/src/pages/WalletAnalysis.jsx` (single page) → calls `Frontend/src/services/api.js`.
