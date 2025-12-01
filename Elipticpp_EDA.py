import os
import pandas as pd
import matplotlib.pyplot as plt

# 1. Data folder
# Use a pre-extracted data folder instead of a zip file.
# Update `base` if your data folder has a different name or path.
base = "Elipticpp_data"
if not os.path.isdir(base):
    raise FileNotFoundError(f"Data folder '{base}' not found. Place the extracted files there or update the path.")

# 2. Load core tables
txs_features = pd.read_csv(os.path.join(base, "txs_features.csv"))
txs_classes  = pd.read_csv(os.path.join(base, "txs_classes.csv"))
txs_edges    = pd.read_csv(os.path.join(base, "txs_edgelist.csv"))

wallets_features = pd.read_csv(os.path.join(base, "wallets_features.csv"))
wallets_classes  = pd.read_csv(os.path.join(base, "wallets_classes.csv"))

# 3. Merge labels
txs = txs_features.merge(txs_classes, on="txId", how="left")
wallets = wallets_features.merge(wallets_classes, on="address", how="left")

print("txs:", txs.shape)
print("wallets:", wallets.shape)

# === BASIC EDA ===

# A. Class distribution
tx_label_counts = txs["class"].value_counts().sort_index()
wallet_label_counts = wallets["class"].value_counts().sort_index()

print("Tx class counts:\n", tx_label_counts)
print("Wallet class counts:\n", wallet_label_counts)

plt.figure()
tx_label_counts.plot(kind="bar")
plt.title("Transaction Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

plt.figure()
wallet_label_counts.plot(kind="bar")
plt.title("Wallet Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# B. Time-step distribution
timestep_counts = txs["Time step"].value_counts().sort_index()
print("Time steps:", timestep_counts.head(), " ... ", timestep_counts.tail())

plt.figure()
timestep_counts.plot(kind="line", marker="o")
plt.title("Transactions per Time Step")
plt.xlabel("Time step")
plt.ylabel("Number of transactions")
plt.tight_layout()
plt.show()

# C. Missing values
tx_missing = txs.isna().mean().sort_values(ascending=False)
print("Top 20 missing (txs):\n", tx_missing.head(20))

wallet_missing = wallets.isna().mean().sort_values(ascending=False)
print("Top 20 missing (wallets):\n", wallet_missing.head(20))

# D. Basic feature stats for a few numeric columns
num_cols_txs = [c for c in txs.columns if c.startswith("Local_feature_")][:5]
print(txs[num_cols_txs].describe().T)

num_cols_wallets = [c for c in wallets.columns if wallets[c].dtype != "object"][:5]
print(wallets[num_cols_wallets].describe().T)
