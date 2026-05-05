# Bitcoin Wallet Risk Analyzer

GNN-based Bitcoin wallet risk classification using REAL-CATS and Elliptic++ datasets.

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 16+

### Run Application

**Backend:**
```bash
cd Backend
python main.py
```

**Frontend:**
```bash
cd Frontend
npm install  # first time only
npm run dev
```

**Access:**
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000/docs

## Training Pipeline

Run the complete pipeline:
```bash
python run_pipeline.py --all
```

Or run individual steps:
```bash
python run_pipeline.py --prepare   # Step 1: Prepare unified dataset
python run_pipeline.py --graphs    # Step 2: Build ego-graphs
python run_pipeline.py --baseline  # Step 3: Train XGBoost baseline
python run_pipeline.py --train     # Step 4: Train GNN model
python run_pipeline.py --evaluate  # Step 5: Evaluate & compare models
```

## Standalone CLI

A self-contained command-line tool for analyzing any Bitcoin wallet without running the full web app.

```bash
cd standalone

# macOS/Linux — one-click setup & run
./setup_and_run.sh 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa

# Windows
setup_and_run.bat 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa

# Or run directly (requires dependencies installed)
python wallet_analyzer.py 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa
python wallet_analyzer.py --no-charts bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh
python wallet_analyzer.py --model /path/to/model.pt <address>
```

- Fetches live transactions from mempool.space
- Builds ego-graph and runs GNN inference
- Generates charts (risk gauge, feature importance, transaction volume) saved to `analyses/<address>/`
- First run auto-installs dependencies into a local venv

## Project Structure

```
final_project/
├── Backend/                    # FastAPI backend
├── Frontend/                   # React frontend
├── standalone/                 # Standalone CLI tool
│   ├── wallet_analyzer.py     # Self-contained analyzer script
│   ├── wallet_analyzer.spec   # PyInstaller build spec
│   ├── setup_and_run.sh       # macOS/Linux one-click installer
│   └── setup_and_run.bat      # Windows one-click installer
├── data/                       # Raw datasets
│   ├── realcats/              # REAL-CATS dataset
│   └── elliptic/              # Elliptic++ dataset
├── src/                        # Source code
│   ├── features/              # Feature extraction
│   ├── graph/                 # Ego-graph construction
│   ├── models/                # GNN model
│   ├── baselines/             # Baseline models (XGBoost)
│   ├── evaluation/            # Evaluation tools
│   └── pipeline_steps/        # One module per run_pipeline.py step
├── scripts/                    # Utility scripts
│   ├── viz/                   # Chart/diagram generation
│   ├── generate_model_report.py
│   └── check_standalone_sync.py  # Verify standalone matches src/
├── notebooks/                  # Jupyter notebooks (EDA only)
├── graph_data/                 # Generated ego-graphs cache (gitignored)
├── outputs/                    # Generated artifacts only — no source code
│   ├── baselines/             # Baseline comparison plots + JSON
│   ├── evaluation/            # Evaluation results, calibration, confusion matrix, etc.
│   ├── gnn_model.pt           # Trained GNN weights (loaded by Backend + standalone)
│   ├── temperature.pt         # Calibration scaler
│   └── gnn_training_history.json
├── docs/                       # Project docs
│   ├── STATUS.md              # Open work + completed work + architecture notes
│   ├── CALL_FLOW.md           # Function-level call flow for the three entry points
│   ├── archive/               # Historical: resolved fix-docs, design reviews
│   └── submissions/           # Course deliverables (slides, lit review)
└── run_pipeline.py            # Master pipeline CLI (thin dispatcher)
```

## Datasets

### REAL-CATS
- 40,032 criminal Bitcoin addresses (behavioral)
- 90,176 benign Bitcoin addresses
- Source: https://github.com/sjdseu/Real-CATS

### Elliptic++
- 822k wallet addresses with transaction features
- Source: https://github.com/git-disl/EllipticPlusPlus

## Model

**OptimalBitcoinGNN** (defined in `src/models/optimal_gnn.py`): 3-layer GATv2 architecture with:
- 12 selected features (from 19)
- Multi-head attention (4→4→2 heads)
- Residual connections
- Dropout regularization

### Saved Model Artifacts

Trained by `src/models/train_optimal.py` (invoked via `run_pipeline.py --train`) and saved to:

| File | Description |
| --- | --- |
| `outputs/gnn_model.pt` | Trained model weights (`state_dict`) — load this for inference |
| `outputs/gnn_checkpoint.pt` | Full checkpoint (model + optimizer + scheduler state) for resuming training |
| `outputs/temperature.pt` | Calibration temperature for probability scaling |
| `outputs/gnn_training_history.json` | Per-epoch training/validation metrics |

Note: although `train_optimal.py` defaults `--save-path` to `optimal_gnn_model.pt`, the pipeline saves under the canonical name `outputs/gnn_model.pt` (referenced by `run_pipeline.py`, `Backend/`, and `standalone/wallet_analyzer.py`).

For the function-level call flow of the offline pipeline, the FastAPI backend, and the standalone CLI, see [`docs/CALL_FLOW.md`](docs/CALL_FLOW.md).

## Technologies

- PyTorch + PyTorch Geometric (GNN)
- FastAPI (Backend)
- React + Vite (Frontend)
- XGBoost (Baseline)
- mempool.space API (Blockchain data)
