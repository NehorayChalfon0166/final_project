# Bitcoin Wallet Risk Analyzer

Graph Neural Network-based Bitcoin wallet risk classification system.

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 16+

### Run

**Terminal 1 - Backend:**
```bash
cd Backend
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm install  # first time only
npm run dev
```

### Access
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
## Usage

1. Enter a Bitcoin wallet address
2. Click "Analyze"
3. View risk classification and confidence score

## Features

- Bitcoin wallet analysis
- Real-time GNN inference
- Risk classification (Criminal/Benign)
- Random wallet generation for testing
- Behavior: transaction_fee, variances, payment patterns
- Structure: input/output slots
- Engineered: flow_ratio, fan_ratio

**Inference Features (8)** - Used for real-time analysis:
- Graph centrality: in_degree, out_degree, pagerank
- Clustering: clustering_coefficient
- Centrality: betweenness, closeness, eigenvector, harmonic

## 📚 Datasets

### REAL-CATS Dataset
Real World Dataset of Cryptocurrency Addresses with Transaction Profiles

- **50,943 criminal addresses** from real-world reports
- **102,178 benign addresses** from exchange customers
- Bitcoin & Ethereum support
- Behavioral features included

**Files:**
- `CB.tsv` - 40,032 criminal Bitcoin addresses
- `BB.tsv` - 90,176 benign Bitcoin addresses
- `CE.tsv` - 12,561 criminal Ethereum addresses  
- `BE.tsv` - 16,020 benign Ethereum addresses

📖 More details: [README_cats.md](README_cats.md)

**Source**: https://github.com/sjdseu/Real-CATS  
**Kaggle**: https://www.kaggle.com/datasets/lvd312393/real-cats

### Elliptic++ Dataset
Graph Network of Bitcoin Blockchain Transactions

- **203k Bitcoin transactions**
- **822k wallet addresses**
- Transaction & actor-level detection
- Temporal features (49 time steps)

📖 More details: [README_eliptic.md](README_eliptic.md)

**Source**: https://github.com/git-disl/EllipticPlusPlus

## 🔧 Training Your Own Model

```bash
cd Backend
python train_model.py
```

**Requirements:**
1. Place REAL-CATS data in `Real_Cats_data/` directory:
   - `BB.tsv` (benign wallets)
   - `CB.tsv` (criminal wallets - will be split into behavioral/non-behavioral)

2. The script will:
   - Load and merge datasets
   - Engineer features
   - Fetch transaction graphs from mempool.space
   - Train GNN model
   - Save to `models/crypto_gnn_model.pt`

**Note**: Fetching transaction data takes time due to API rate limits.

## 📁 Project Structure

```
final_project/
├── Backend/
│   ├── main.py                      # FastAPI application
│   ├── train_model.py               # Model training script
│   ├── routes/
│   │   ├── wallet.py                # Wallet CRUD + Analysis endpoints
│   │   ├── transactions.py          # Transaction endpoints
│   │   ├── health.py                # Health check endpoints
│   │   └── utils/
│   │       └── model_fit_utils.py   # Core ML utilities
│   └── README.md                    # API documentation
├── models/
│   ├── model_pipeline.ipynb         # Training notebook
│   └── crypto_gnn_model.pt          # Trained model (after training)
├── eda/
│   ├── Real-CATS-EDA-Clean.ipynb   # REAL-CATS analysis
│   └── Elipticpp_EDA.ipynb         # Elliptic++ analysis
├── Elipticpp_Data/                  # Elliptic++ dataset files
├── Real_Cats_data/                  # REAL-CATS dataset files
└── requirements.txt                 # Python dependencies
```

## 🔍 Example Usage

### Analyze a Wallet via API

```python
import requests

# Analyze Bitcoin wallet
address = "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
response = requests.post(
    f"http://localhost:8000/api/v1/analyze/{address}",
    params={"model_path": "../models/crypto_gnn_model.pt"}
)

result = response.json()
print(f"Risk Score: {result['risk_score']}")
print(f"Nodes: {result['nodes_count']}, Edges: {result['edges_count']}")
```

### Using Python Directly

```python
from Backend.routes.utils.model_fit_utils import analyze_wallet_pipeline

result = analyze_wallet_pipeline(
    wallet_address="1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
    model_path="models/crypto_gnn_model.pt"
)

print(result)
```

## 🛠️ Technologies

- **FastAPI** - Modern web framework
- **PyTorch** - Deep learning framework
- **PyTorch Geometric** - Graph neural networks
- **scikit-learn** - Feature preprocessing
- **pandas** - Data manipulation
- **NetworkX** - Graph analysis
- **mempool.space API** - Live blockchain data

## 📊 Performance

The GNN model achieves:
- Real-time inference (< 5 seconds per wallet)
- Scalable graph representation
- Attention-based feature learning
- Support for both labeled and unlabeled nodes

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Multi-blockchain support (Ethereum, other chains)
- Enhanced feature engineering
- Real-time monitoring dashboard
- Database integration (PostgreSQL, MongoDB)
- Caching layer for analyzed wallets

## 📄 License

This project uses public datasets:
- REAL-CATS: Check repository for license
- Elliptic++: Check repository for license

## 📞 Contact

For questions or collaborations:
- GitHub: [@NehorayChalfon0166](https://github.com/NehorayChalfon0166)

## 🙏 Acknowledgments

- REAL-CATS Dataset creators
- Elliptic++ Dataset creators
- mempool.space for blockchain API access