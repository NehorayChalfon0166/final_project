# Backend API - Cryptocurrency Wallet Risk Analysis

FastAPI backend for real-time cryptocurrency wallet analysis and risk classification using Graph Neural Networks.

## 🚀 Quick Start

### Install Dependencies

```bash
pip install -r ../requirements.txt
```

### Run the Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`

### Access Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📡 API Endpoints

### Wallet Analysis

#### `POST /api/v1/analyze/{address}`
Analyze a cryptocurrency wallet address for risk assessment.

**Parameters:**
- `address` (path) - Wallet address to analyze (Bitcoin, Ethereum, or hex format)
- `model_path` (query, optional) - Path to trained model file
- `save_to_db` (query, optional, default=true) - Save results to database

**Response:**
```json
{
  "wallet_address": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
  "status": "success",
  "nodes_count": 150,
  "edges_count": 423,
  "graph_data": {
    "x_shape": [150, 8],
    "y_shape": [150],
    "edge_index_shape": [2, 423],
    "edge_attr_shape": [423, 1]
  },
  "risk_score": 0.73,
  "prediction": [0.27, 0.73]
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/analyze/1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa?model_path=../models/crypto_gnn_model.pt"
```

### Wallet Management (CRUD)

#### `GET /api/v1/wallets`
List all stored wallet records.

**Response:**
```json
[
  {
    "id": 1,
    "address": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
    "is_valid": true,
    "balance": 0.0,
    "risk_score": 0.73,
    "last_analyzed": "2025-12-28T10:30:00",
    "created_at": "2025-12-28T09:00:00"
  }
]
```

#### `GET /api/v1/wallets/{wallet_id}`
Get a specific wallet by ID.

#### `POST /api/v1/wallets/validate`
Validate a wallet address format.

**Request:**
```json
{
  "address": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
  "balance": 0.0
}
```

**Response:**
```json
{
  "address": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
  "is_valid": true,
  "message": "Valid wallet address"
}
```

#### `POST /api/v1/wallets`
Create a new wallet record.

#### `PUT /api/v1/wallets/{wallet_id}`
Update an existing wallet.

#### `DELETE /api/v1/wallets/{wallet_id}`
Delete a wallet record.

### Health & Status

#### `GET /api/v1/health`
Check API health status.

#### `GET /api/v1/ping`
Ping the API.

## 🧠 Model Utilities

The core ML functionality is in `routes/utils/model_fit_utils.py`.

### Real-time Analysis Functions

- `fetch_edges_mempool_directed(wallet_address)` - Fetch transaction edges from mempool.space
- `process_and_save_tensors(wallet_address, df_edges)` - Build graph tensors
- `analyze_wallet_pipeline(wallet_address, model_path)` - Complete analysis pipeline

### Training Functions

- `load_and_label_datasets()` - Load REAL-CATS datasets
- `merge_datasets()` - Merge benign/criminal data
- `perform_feature_engineering()` - Create engineered features
- `train_model()` - Train GNN model

### Model Architecture

```python
class CryptoGNN(torch.nn.Module):
    """Graph Attention Network for wallet risk classification"""
    - GATv2Conv Layer 1 (2 heads)
    - Dropout (0.3)
    - GATv2Conv Layer 2 (1 head)
    - Linear Classifier (2 classes)
```

## 🎓 Training Your Model

```bash
python train_model.py
```

Requires REAL-CATS data in `../Real_Cats_data/` directory.

## 📊 Feature Sets

### Basic Features (8) - Real-time Inference
- in_degree, out_degree, pagerank
- clustering_coefficient, betweenness, closeness
- eigenvector, harmonic

### Full Features (20) - Training
- Volume: balance, total_received_USD, total_sent_USD
- Velocity: lifetime, transaction_number, activity metrics
- Behavior: fees, variances, payment patterns
- Engineered: flow_ratio, fan_ratio

## 📚 Related Documentation

- Main README: [../README.md](../README.md)
- REAL-CATS Dataset: [../README_cats.md](../README_cats.md)
- Elliptic++ Dataset: [../README_eliptic.md](../README_eliptic.md)
