import pandas as pd
import os

# Paths (based on your project tree)
cb_path = os.path.join("Real_Cats_data", "CB.tsv")
wallets_path = os.path.join("Elipticpp_Data", "wallets_classes.csv")

# Load data
cb = pd.read_csv(cb_path, sep="\t")
wallets = pd.read_csv(wallets_path)

# Sanity check
if "address" not in cb.columns:
    raise KeyError(f"'address' column missing in CB.tsv. Columns: {cb.columns.tolist()}")
if "address" not in wallets.columns or "class" not in wallets.columns:
    raise KeyError(f"'address' or 'class' missing in wallets_classes.csv. Columns: {wallets.columns.tolist()}")

# All wallets in CB are illicit by definition
cb_illicit_addresses = cb["address"].astype(str).unique()
print(f"Total CB (ground-truth illicit) wallets: {len(cb_illicit_addresses)}")

# Merge to see how Elliptic labels them
merged = (
    pd.DataFrame({"address": cb_illicit_addresses})
    .merge(wallets[["address", "class"]], on="address", how="left")
)

# Map Elliptic numeric classes -> labels
class_mapping = {1: "Illicit", 2: "Licit", 3: "Unknown"}
merged["class_label"] = merged["class"].map(class_mapping)

# How many CB wallets appear in Elliptic at all?
present_in_elliptic = merged[merged["class"].notna()]
missing_in_elliptic = merged[merged["class"].isna()]

print(f"CB wallets found in Elliptic: {len(present_in_elliptic)}")
print(f"CB wallets NOT found in Elliptic: {len(missing_in_elliptic)}")

# Distribution of Elliptic labels for CB-illicit wallets
print("\nElliptic labels for CB-illicit wallets:")
print(present_in_elliptic["class_label"].value_counts(dropna=False))

# Explicit counts of (in)consistency vs ground truth
correct_illicit = present_in_elliptic[present_in_elliptic["class"] == 1]
incorrect_licit = present_in_elliptic[present_in_elliptic["class"] == 2]
incorrect_unknown = present_in_elliptic[present_in_elliptic["class"] == 3]

print(f"\nCorrectly labeled as illicit in Elliptic (class=1): {len(correct_illicit)}")
print(f"Labeled as licit in Elliptic (class=2): {len(incorrect_licit)}")
print(f"Labeled as unknown in Elliptic (class=3): {len(incorrect_unknown)}")

# Show a few examples of mismatches
print("\nExample CB-illicit but Elliptic-LICIT wallets:")
print(incorrect_licit["address"].head(10).tolist())

print("\nExample CB-illicit but Elliptic-UNKNOWN wallets:")
print(incorrect_unknown["address"].head(10).tolist())
