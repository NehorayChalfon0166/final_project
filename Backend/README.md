# Backend API - Cryptocurrency Wallet Risk Analysis

FastAPI backend for real-time cryptocurrency wallet analysis and risk classification using Graph Neural Networks (GNN).

## 🎯 Features

- **Real-time Wallet Analysis**: Fetch live transaction data from blockchain
- **GNN-Based Risk Assessment**: Binary classification (Criminal/Benign)
- **Transaction Graph Construction**: Build directed graphs from wallet transactions
- **Feature Engineering**: Compute 12 behavioral features from transaction patterns
- **RESTful API**: Easy integration with frontend applications

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Virtual environment activated
- Dependencies installed from `requirements.txt`

### Install Dependencies

```bash
# From project root
pip install -r requirements.txt
```

### Run the Server

```bash
cd Backend
python main.py
```

The API will be available at:
- **Server**: https://CryptoTrace.cs.bgu.ac.il
- **Swagger UI**: https://CryptoTrace.cs.bgu.ac.il/docs
- **ReDoc**: https://CryptoTrace.cs.bgu.ac.il/redoc

## 📡 API Endpoints

### 🔍 Wallet Analysis

#### `GET /api/v1/analyze/{address}`

Analyze a cryptocurrency wallet address for risk assessment.

**Parameters:**
- `address` (path, required) - Wallet address to analyze
  - Bitcoin: `1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa`
  - Ethereum: `0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb`
  - Generic hex: 40-64 character hex string
- `model_path` (query, optional) - Path to trained model (default: `../outputs/gnn_model.pt`)

**Response:**
```json
{
  "wallet_address": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
  "status": "success",
  "nodes_count": 50,
  "edges_count": 51,
  "graph_data": {
    "x_shape": [50, 12],
    "y_shape": [50],
    "edge_index_shape": [2, 51],
    "edge_attr_shape": [51, 1]
  },
  "classification": "benign",
  "prediction": [0.544, 0.456],
  "risk_score": 0.456,
  "confidence": 0.544,
  "message": "Wallet classified as BENIGN with 54.4% confidence"
}
```

**Classification:**
- `"benign"` - Risk score ≤ 0.5 (Low risk)
- `"criminal"` - Risk score > 0.5 (High risk)

**Example cURL:**
```bash
curl -X GET "http://localhost:8000/api/v1/analyze/1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
```

**Example Response Fields:**
- `classification` - Final verdict: "criminal" or "benign"
- `risk_score` - Probability of criminal activity (0.0 to 1.0)
- `confidence` - Confidence in classification (0.0 to 1.0)
- `prediction` - [prob_benign, prob_criminal]

### ❤️ Health Check

#### `GET /api/v1/health`

Check API health status.

**Response:**
```json
{
  "status": "healthy"
```

#### `GET /api/v1/ping`

Simple ping endpoint to test connectivity.

**Response:**
```json
{
  "message": "pong"
}
```

### 💰 Transaction Endpoints

#### `GET /api/v1/transactions`
List all transactions (in-memory demo).

#### `POST /api/v1/transactions`
Create a new transaction record.

#### `GET /api/v1/transactions/{tx_id}`
Get specific transaction details.

## 🏗️ Project Structure

```
Backend/
├── main.py                 # FastAPI application entry point
├── openapi.yaml           # OpenAPI 3.0 specification
├── README.md              # This file
├── routes/
│   ├── __init__.py
│   ├── health.py          # Health check endpoints
│   ├── wallet.py          # Wallet analysis endpoint
│   ├── transactions.py    # Transaction management
│   └── utils/
│       ├── __init__.py
│       └── model_fit_utils.py  # ML utilities & GNN model
```

## 🧠 ML Pipeline

### Analysis Flow

1. **Fetch Transactions** - Pull live data from mempool.space API
2. **Build Graph** - Construct directed transaction graph
3. **Feature Engineering** - Compute 18 REAL-CATS features
4. **Preprocessing** - Log scaling & standardization
5. **GNN Inference** - Run Graph Attention Network model
6. **Classification** - Binary output (Criminal/Benign)

### Model Architecture (OptimalBitcoinGNN)

```
3-layer GATv2 with residual connections:
    - Input: 12 node features
    - GATv2Conv Layer 1 (4 attention heads, 64 hidden → 256 output)
    - GATv2Conv Layer 2 (4 attention heads, 256 → 256 + residual)
    - GATv2Conv Layer 3 (2 attention heads, 256 → 64)
    - Classifier: Linear(64 → 32 → 2)
```

### Feature Set (12 Features)

Selected from REAL-CATS and Elliptic++ datasets (7 correlated features removed):
- **Temporal**: lifetime_seconds, activity_rate, blocks_btwn_txs_mean
- **Transaction Flow**: in_out_balance, send_receive_ratio, total_txs
- **Fee Behavior**: fee_per_tx, fee_share_mean
- **Amount Characteristics**: avg_tx_size, tx_size_range, max_sent, max_received

## 🔧 Configuration

### Environment Variables

Create a `.env` file (optional):
```bash
MODEL_PATH=../models/crypto_gnn_model.pt
API_HOST=127.0.0.1
API_PORT=8000
DEBUG=True
USE_HTTPS=False
SSL_KEYFILE=./privkey.pem
SSL_CERTFILE=./fullchain.pem
```

For local development, keep `API_HOST=127.0.0.1` and `USE_HTTPS=False`. For production TLS termination directly in uvicorn, set `USE_HTTPS=True` and point `SSL_KEYFILE` and `SSL_CERTFILE` to your certificate files. If you are behind a reverse proxy, keep `USE_HTTPS=False` and let the proxy terminate TLS.

### Model Path

Default model location: `../outputs/gnn_model.pt`

## 📝 Development

### Running in Development Mode

```bash
# With auto-reload
python main.py

# Or with uvicorn directly
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### Code Quality

Recommended tools:
```bash
# Format code
black .

# Type checking
mypy .

# Linting
pylint routes/
```

## 🧪 Testing

### Test with cURL

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Analyze wallet
curl http://localhost:8000/api/v1/analyze/1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa
```

### Test with Postman

Import the OpenAPI spec from `openapi.yaml` or use the Swagger UI at `/docs`.

## 📚 Additional Resources

- **Main Project README**: [../README.md](../README.md)
- **REAL-CATS Dataset**: https://github.com/sjdseu/Real-CATS
- **Elliptic++ Dataset**: https://github.com/git-disl/EllipticPlusPlus

## ⚠️ Important Notes

- API rate limit: 1 request/second for mempool.space
- Analysis time: ~2-5 seconds per wallet (depends on transaction count)
- Model requires 12 input features (selected from REAL-CATS + Elliptic++ data)
- Classification threshold: risk_score > 0.5 = criminal

## 📄 License

Part of the Cryptocurrency Wallet Risk Analysis project.
