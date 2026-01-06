"""
Feature Configuration
======================
Defines the feature sets for both pipeline options.

Option A (Conservative): 14 features that are accurate for BOTH datasets
Option B (Recommended):  20 features for REAL-CATS, 14 for Elliptic++ with masking
"""

# =============================================================================
# OPTION A: CONSERVATIVE (14 Features)
# =============================================================================
# Only features that can be accurately calculated for BOTH datasets
# No API calls required, no approximations

CONSERVATIVE_SHARED = [
    'total_txs',           # transaction_number | total_txs
    'total_sent_btc',      # total_sent_BTC | btc_sent_total
    'total_received_btc',  # total_received_BTC | btc_received_total
    'total_fees',          # transaction_fee | fees_total
    'lifetime_seconds',    # lifetime | lifetime_in_blocks * 600
    'num_send_txs',        # payment_transactions | num_txs_as_sender
    'num_receive_txs',     # receipt_transactions | num_txs_as receiver
    'max_sent',            # max_sent_amount | btc_sent_max
    'min_sent',            # min_sent_amount | btc_sent_min
    'max_received',        # max_received_amount | btc_received_max
    'min_received',        # min_received_amount | btc_received_min
]

CONSERVATIVE_CALCULABLE = [
    # These can be accurately calculated from existing columns in BOTH datasets
    'blocks_btwn_txs_mean',  # Elliptic++ has it; REAL-CATS: lifetime/(tx-1)/600
    'fee_share_mean',        # Elliptic++ has it; REAL-CATS: fees/total_transacted
    'avg_tx_size',           # Elliptic++ has it; REAL-CATS: total/tx_count
]

CONSERVATIVE_ENGINEERED = [
    # Derived from shared features - accurate for both
    'send_receive_ratio',    # total_sent / total_received
    'activity_rate',         # total_txs / lifetime
    'fee_per_tx',            # total_fees / total_txs
    'tx_size_range',         # max_sent - min_sent
    'in_out_balance',        # num_receive_txs / num_send_txs
]

# Total Conservative Features: 11 + 3 + 5 = 19 (but some overlap, actual = 14 base + 5 engineered)
CONSERVATIVE_ALL = CONSERVATIVE_SHARED + CONSERVATIVE_CALCULABLE + CONSERVATIVE_ENGINEERED


# =============================================================================
# OPTION B: RECOMMENDED (Full for REAL-CATS, Partial for Elliptic++)
# =============================================================================

# Features unique to REAL-CATS (available directly, need API for Elliptic++)
REALCATS_UNIQUE = [
    'fee_variance',           # transaction_fee_Variance - detects automated vs manual
    'sent_variance_btc',      # sent_Variance_BTC - identifies fixed-amount patterns
    'received_variance_btc',  # received_Variance_BTC - detects ransom patterns
    'output_slots',           # total_output_slots - reveals mixing behavior
]

# Features unique to Elliptic++ (available directly, need API for REAL-CATS)
ELLIPTIC_UNIQUE = [
    'blocks_btwn_txs_mean',   # temporal fingerprint (also calculable for REAL-CATS)
    'unique_addresses',       # network diversity - NEEDS API for REAL-CATS
    'repeated_addresses',     # repeated interactions - NEEDS API for REAL-CATS
    'fee_share_mean',         # fee behavior (also calculable for REAL-CATS)
    'avg_tx_size',            # average tx size (also calculable for REAL-CATS)
]

# Features that can be calculated for REAL-CATS (subset of ELLIPTIC_UNIQUE)
ELLIPTIC_CALCULABLE_FOR_REALCATS = [
    'blocks_btwn_txs_mean',
    'fee_share_mean',
    'avg_tx_size',
]

# Features that REQUIRE API for REAL-CATS
ELLIPTIC_NEEDS_API_FOR_REALCATS = [
    'unique_addresses',
    'repeated_addresses',
]

# Engineered features (same as conservative)
RECOMMENDED_ENGINEERED = CONSERVATIVE_ENGINEERED

# Full feature set for REAL-CATS (20 features)
REALCATS_FULL_FEATURES = (
    CONSERVATIVE_SHARED +
    REALCATS_UNIQUE +
    ELLIPTIC_CALCULABLE_FOR_REALCATS +
    RECOMMENDED_ENGINEERED
)

# Feature set for Elliptic++ without API (14 features)
ELLIPTIC_PARTIAL_FEATURES = (
    CONSERVATIVE_SHARED +
    ELLIPTIC_UNIQUE +  # All available directly
    RECOMMENDED_ENGINEERED
)

# =============================================================================
# COLUMN MAPPINGS
# =============================================================================

# Maps unified column name to (realcats_column, elliptic_column)
COLUMN_MAPPING = {
    # Shared features
    'total_txs': ('transaction_number', 'total_txs'),
    'total_sent_btc': ('total_sent_BTC', 'btc_sent_total'),
    'total_received_btc': ('total_received_BTC', 'btc_received_total'),
    'total_fees': ('transaction_fee', 'fees_total'),
    'lifetime_seconds': ('lifetime', 'lifetime_in_blocks'),  # Elliptic needs *600
    'num_send_txs': ('payment_transactions', 'num_txs_as_sender'),
    'num_receive_txs': ('receipt_transactions', 'num_txs_as receiver'),
    'max_sent': ('max_sent_amount', 'btc_sent_max'),
    'min_sent': ('min_sent_amount', 'btc_sent_min'),
    'max_received': ('max_received_amount', 'btc_received_max'),
    'min_received': ('min_received_amount', 'btc_received_min'),

    # REAL-CATS unique
    'fee_variance': ('transaction_fee_Variance', None),
    'sent_variance_btc': ('sent_Variance_BTC', None),
    'received_variance_btc': ('received_Variance_BTC', None),
    'output_slots': ('total_output_slots', None),

    # Elliptic++ unique
    'blocks_btwn_txs_mean': (None, 'blocks_btwn_txs_mean'),
    'unique_addresses': (None, 'transacted_w_address_total'),
    'repeated_addresses': (None, 'num_addr_transacted_multiple'),
    'fee_share_mean': (None, 'fees_as_share_mean'),
    'avg_tx_size': (None, 'btc_transacted_mean'),
}

# =============================================================================
# CONSTANTS
# =============================================================================

SECONDS_PER_BLOCK = 600  # Bitcoin block time ~10 minutes
EPSILON = 1e-10  # Small value to avoid division by zero
