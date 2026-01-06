"""
Graph Pipeline Configuration
============================
Paths, API settings, and feature definitions.
"""
import os

# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURE_DIR = os.path.join(PROJECT_ROOT, 'src', 'features', 'output')
GRAPH_DATA_DIR = os.path.join(PROJECT_ROOT, 'graph_data')

# Dataset paths
TRAIN_DATASET_PATH = os.path.join(FEATURE_DIR, 'train_dataset.csv')
TEST_DATASET_PATH = os.path.join(FEATURE_DIR, 'test_dataset.csv')

# Cache directories
CACHE_DIR = os.path.join(GRAPH_DATA_DIR, 'cache')
GRAPHS_DIR = os.path.join(GRAPH_DATA_DIR, 'graphs')
METADATA_DIR = os.path.join(GRAPH_DATA_DIR, 'metadata')


# =============================================================================
# API CONFIGURATION
# =============================================================================

MEMPOOL_API_BASE = "https://mempool.space/api"
MEMPOOL_RATE_LIMIT = 10      # requests per second
CONCURRENT_REQUESTS = 8      # parallel async requests
REQUEST_TIMEOUT = 15         # seconds
MAX_RETRIES = 3
RETRY_DELAY = 2.0            # seconds between retries


# =============================================================================
# FEATURE COLUMNS (12 selected features - removed 7 highly correlated)
# =============================================================================
#
# REMOVED (with justification):
# - total_sent_btc_log: Correlated with total_received_btc (r=0.9752)
# - total_received_btc_log: Covered by send_receive_ratio and max values
# - num_receive_txs_log: Correlated with total_txs (r=0.9897)
# - num_send_txs_log: Correlated with total_txs, covered by in_out_balance
# - total_fees_log: Covered by fee_per_tx (normalized, more informative)
# - min_sent_log: Low discriminative value, tx_size_range captures range
# - min_received_log: Same reasoning as min_sent

FEATURE_COLUMNS = [
    # Top discriminative features
    'lifetime_seconds_log',       # Highest discriminative power (diff: 1.998)
    'activity_rate_log',          # 2nd highest (diff: 0.918)
    'in_out_balance_log',         # 3rd highest (diff: 0.651)
    'total_txs_log',              # Overall activity level

    # Behavioral patterns
    'send_receive_ratio_log',     # Flow direction
    'fee_per_tx_log',             # Automated vs manual patterns
    'blocks_btwn_txs_mean_log',   # Temporal pattern
    'fee_share_mean_log',         # Fee behavior

    # Transaction characteristics
    'avg_tx_size_log',            # Average transaction size
    'tx_size_range_log',          # Range indicates fixed vs varied amounts
    'max_sent_log',               # Large fund movements
    'max_received_log',           # Large incoming amounts
]

NUM_NODE_FEATURES = len(FEATURE_COLUMNS)  # 12
NUM_EDGE_FEATURES = 3  # amount_log, direction, timestamp_norm


# =============================================================================
# TIMESTAMP NORMALIZATION
# =============================================================================

# Bitcoin genesis block: 2009-01-03
TIMESTAMP_MIN = 1231006505

# Future buffer: ~2030-01-01
TIMESTAMP_MAX = 1893456000


# =============================================================================
# BATCH PROCESSING
# =============================================================================

BATCH_SIZE = 100              # Process N addresses at a time
CHECKPOINT_INTERVAL = 500     # Save progress every N addresses
