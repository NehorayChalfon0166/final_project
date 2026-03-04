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
│   └── evaluation/            # Evaluation tools
├── scripts/                    # Utility scripts
│   ├── evaluate_model.py      # Model evaluation on test set
│   ├── check_model.py         # Model inspection
│   └── build_remaining_graphs.py
├── notebooks/                  # Jupyter notebooks
├── graph_data/                 # Generated ego-graphs
├── outputs/                    # All outputs
│   ├── baseline/              # XGBoost results
│   ├── evaluation/            # Evaluation results & analysis
│   ├── gnn_model.pt           # Trained GNN model
│   └── gnn_training_history.json
└── run_pipeline.py            # Master pipeline script
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

**OptimalBitcoinGNN**: 3-layer GATv2 architecture with:
- 12 selected features (from 19)
- Multi-head attention (4→4→2 heads)
- Residual connections
- Dropout regularization

## Technologies

- PyTorch + PyTorch Geometric (GNN)
- FastAPI (Backend)
- React + Vite (Frontend)
- XGBoost (Baseline)
- mempool.space API (Blockchain data)
