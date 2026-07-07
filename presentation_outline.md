# Final Presentation Outline — 12 minutes
**Bitcoin Wallet Risk Analyzer: Detecting Criminal Wallets with GNNs** (Group 4)

~12 slides ≈ 1 min/slide. Slide 1 (title + credits: team + professor) is already done.

---

## Slide 2 — The Problem (1 min) ⭐ strong open
**Message: "Billions in crypto crime hide in plain sight on a public ledger."**
- One striking visual: a dark slide with a single big stat (e.g., illicit crypto volume per year — pull a current figure from Chainalysis Crypto Crime Report) over a faded Bitcoin transaction-graph image.
- One sentence: every transaction is public, yet criminal wallets are nearly impossible to spot manually among hundreds of millions of addresses.
- No bullets. Speak over the image.

## Slide 3 — Motivation (1 min)
- Why it matters: ransomware, scams, money laundering all cash out through wallets; exchanges and regulators (AML/KYC) need automated screening.
- Why it's hard: anonymity, scale (822k+ addresses in our data alone), criminals mimic normal behavior.
- Visual: 3 icons (ransomware / exchange / regulator), minimal text.

## Slide 4 — Project Goal & Background (1 min)
- Goal in one line: "Classify a Bitcoin wallet as criminal or benign from its transaction behavior and graph neighborhood — and serve it as a usable tool."
- Quick background: what a wallet/address is, transactions form a graph → this is why graphs matter (sets up the GNN).
- Visual: tiny ego-graph sketch — a wallet node with its neighbors.

## Slide 5 — Existing Approaches & Our Uniqueness (1 min)
- Existing: manual blockchain forensics (Chainalysis-style, expensive, closed), classic ML on hand-crafted features (ignores graph structure), prior academic work on Elliptic (transaction-level, not wallet-level).
- Our edge (3 points max):
  1. **Wallet-level** detection on two combined datasets (REAL-CATS 40k criminal + 90k benign addresses; Elliptic++ features).
  2. **Graph structure**: GATv2 attention over ego-graphs, not just per-wallet features.
  3. **End-to-end system**: live lookup of any address via mempool.space → risk score in a web app.

## Slide 6 — Our Solution: Pipeline (1 min)
- Diagram: data → feature extraction (12 selected of 19 features) → ego-graph construction → GNN → calibrated risk score → API → UI.
- Use existing image: `outputs/poster/gnn_training_pipeline.png` or `bitcoin_wallet_workflow.png`.
- One sentence per stage, spoken not written.

## Slide 7 — The Model (1 min)
- OptimalBitcoinGNN: 3-layer **GATv2**, multi-head attention (4→4→2), residual connections, dropout, temperature calibration for trustworthy probabilities.
- Why attention: not all neighbors matter equally — the model learns *which* transactions are suspicious context.
- Use existing image: `outputs/poster/gnn_architecture.png`. Keep math off the slide.

## Slide 8 — Demo (video, ~1.5 min)
- 60–90 sec screen recording, **no live demo**. Record only the interesting parts:
  1. Paste a real address → risk score + confidence appears.
  2. The ego-graph / neighborhood visualization (if the frontend shows it).
  3. One clearly criminal vs. one benign example, side by side.
- Skip menus, login, navigation. Add captions instead of narrating UI clicks.

## Slide 9 — Results: Evaluation (1 min)
- Headline: **89.4% accuracy, ROC-AUC 0.959, F1 0.87** on 11,880 held-out wallets (4,740 criminal).
- Visual: ROC curve (`outputs/poster/roc_curve_light.png`) + confusion matrix (TP 4190 / TN 6432 / FP 708 / FN 550).
- Say what FN means in practice: ~11.6% of criminal wallets missed — and that recall (88.4%) is the metric that matters for screening.

## Slide 10 — Results: vs. Baselines (1 min)
- Bar chart of 3 models (`outputs/poster/baseline_comparison_light.png`):
  - XGBoost (features only): 82.6% acc, recall **67.3%**
  - Basic GCN: 70.1% acc
  - **Our GATv2: 89.4% acc, recall 88.4%**
- Key takeaway sentence: "Graph structure + attention adds +21 points of recall over a strong feature-only baseline — the criminals XGBoost misses are the ones hiding behind normal-looking features."
- If you ran error analysis (gnn_false_negatives.csv): one sentence on what the model still misses.

## Slide 11 — Limitations & Future Work (0.5–1 min)
- 2–3 honest items: dataset labels are imperfect/dated; behavior drift (criminals adapt); ego-graph depth limits; no user-study evaluation.
- Future: temporal GNNs, larger graphs, exchange integration.

## Slide 12 — Summary & Conclusions (1 min) ⭐ strong close
- 3 takeaways: (1) graph structure is essential for catching criminal wallets, (2) attention-based GNN beats strong baselines by a wide recall margin, (3) it works as a real tool, not just a notebook.
- End by mirroring the opening message: **"The blockchain is public — we built the lens that makes crime on it visible."**
- Final slide stays up during Q&A: title + one-line result + names.

---

## Timing budget (12:00)
| Section | Slides | Time |
|---|---|---|
| Open: problem + motivation | 2–3 | 2:00 |
| Goal + uniqueness | 4–5 | 2:00 |
| Solution + model | 6–7 | 2:00 |
| Demo video | 8 | 1:30 |
| Results | 9–10 | 2:00 |
| Limitations + summary | 11–12 | 1:30 |
| Buffer | — | 1:00 |

## Tips
- Rehearse the open and close word-for-word; improvise the middle.
- Max ~20 words per slide outside of figures.
- Embed the demo video in the pptx (don't rely on internet/alt-tab).
- Verify the crypto-crime stat on slide 2 is current before presenting.
