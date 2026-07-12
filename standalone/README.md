# Standalone Wallet Analyzer

A single, self-contained script that classifies a Bitcoin wallet as **criminal**
or **benign** using the project's trained GATv2 GNN — no manual setup required.

When you run it, it will automatically:

1. **Install every library it needs** (`numpy`, `torch`, `torch-geometric`, `requests`).
2. **Load the trained model** — found locally if you're inside the project repo,
   otherwise downloaded from GitHub on first run.
3. **Fetch the wallet's live history** from mempool.space.
4. **Print a report** containing:
   - a **risk percentage** (chance the wallet is criminal),
   - **money in / money out**,
   - **total money currently in the wallet** (BTC and USD),
   - the **top 3 features** that drove the model's decision.

## One-click run

- **macOS** — double-click **`run.command`** in Finder.
  (First time only: right-click → *Open* to bypass Gatekeeper.)
- **Windows** — double-click **`run.bat`**.

These launchers build an isolated environment in `.venv/` so nothing touches
your system Python.

## Command line

```bash
python3 wallet_analyzer.py                          # prompts for an address
python3 wallet_analyzer.py bc1qxy...address...       # analyze directly
python3 wallet_analyzer.py 1A1zP1eP...  --json       # machine-readable JSON
```

**Requirements:** Python 3.9+ and an internet connection (for the blockchain
data, and for downloading libraries/model on first run).

## Example

```
================================================================
  BITCOIN WALLET RISK REPORT
================================================================
  Wallet    : 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa
  Analyzed  : 2026-07-11 12:00:00 UTC

  VERDICT   : ✓  BENIGN
  Risk      : 4.3%  chance this wallet is criminal
  Confidence: 91.4%

  ----------------------------------------------------------
  MONEY
    Money in  (total received) : 99.12345678 BTC  ($6,357,...)
    Money out (total sent)     : 12.00000000 BTC  ($769,...)
    Current balance            : 87.12345678 BTC  ($5,588,...)
    Total transactions         : 1,042

  ----------------------------------------------------------
  TOP 3 FEATURES DRIVING THIS DECISION
    1. Wallet lifetime (active time span)
       contribution: 31.2%
    2. Activity rate (transactions per day)
       contribution: 18.7%
    3. Total number of transactions
       contribution: 12.4%

  ----------------------------------------------------------
  Analyzed graph: 51 nodes · 88 edges · 50 neighbors
================================================================
```

> **Note:** for speed the analyzer reads up to ~75 recent transactions when
> building the graph. The money totals above (received / sent / balance) come
> from mempool.space's full on-chain address stats, so they are always complete.
