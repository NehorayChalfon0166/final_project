import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =======================
# Load combined wallets file
# =======================
path = os.path.join("Elipticpp_Data", "wallets_features_classes_combined.csv")
df = pd.read_csv(path)

print("Loaded dataframe:", df.shape)
print("Columns:", len(df.columns))

# =======================
# Select numeric features
# =======================
# Exclude address and non-numeric columns automatically
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

if "class" in numeric_cols:
    feature_cols = [c for c in numeric_cols if c != "class"]
else:
    feature_cols = numeric_cols

print(f"Numeric feature columns: {len(feature_cols)}")

# =======================
# 1. Feature–Feature Correlation
# =======================
corr_matrix = df[feature_cols].corr(method="pearson")

plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix,
            cmap="coolwarm",
            center=0,
            square=False,
            cbar_kws={"shrink": 0.6})
plt.title("Feature–Feature Correlation (Pearson)")
plt.tight_layout()
plt.show()

# Report highly correlated pairs
THRESHOLD = 0.80
print(f"\nHighly correlated features (|corr| >= {THRESHOLD}):")
reported = set()

for i, c1 in enumerate(feature_cols):
    for j, c2 in enumerate(feature_cols):
        if j <= i:
            continue
        corr_val = corr_matrix.loc[c1, c2]
        if abs(corr_val) >= THRESHOLD:
            pair = tuple(sorted([c1, c2]))
            if pair not in reported:
                print(f"{c1:35s} <-> {c2:35s} : {corr_val:.3f}")
                reported.add(pair)

# =======================
# 2. Feature–Label Correlation (class)
# =======================
if "class" in df.columns:
    corr_with_label = (
        df[feature_cols + ["class"]]
        .corr(method="pearson")["class"]
        .drop("class")
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )

    print("\nTop 20 features most correlated with class:")
    print(corr_with_label.head(20))

    # Plot top 20
    plt.figure(figsize=(10, 6))
    corr_with_label.head(20).sort_values().plot(kind="barh")
    plt.xlabel("Correlation with class (Pearson)")
    plt.title("Top 20 Feature–Class Correlations")
    plt.tight_layout()
    plt.show()
else:
    print("Column 'class' not found — skipping feature–label correlation.")
