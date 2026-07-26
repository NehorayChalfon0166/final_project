# Speaker Notes — Slides 9–11 (~3.5 min)

## Slide 9 — Results: Evaluation (~80 sec)

"So, does it actually work? We evaluated on 11,880 wallets the model had never seen — about 4,700 of them criminal.

The headline numbers: 89.4% accuracy and a ROC-AUC of 0.959. But for this problem, the number we care about most is recall — when a wallet really is criminal, do we catch it? Our recall is 88.4%.

Look at the confusion matrix. Out of 4,740 criminal wallets, we missed only 550. And on the other side, only 708 benign wallets out of more than seven thousand were wrongly flagged.

Why does this balance matter? Think about who uses this. An exchange screening deposits would rather double-check a few hundred false alarms than let thousands of criminal wallets through. A missed criminal is expensive; a false alarm is just a short manual review.

One more thing — these probabilities are temperature-calibrated. When the model says 90% risk, it's right about 90% of the time. That's what makes the score usable for real decisions, not just rankings."

## Slide 10 — Does Graph Structure Matter? (~80 sec)

"Now the key question of the project: did the graph actually buy us anything? Maybe a simple model on the same features would do just as well.

So we ran a controlled comparison — same 12 features, same test set, three models.

XGBoost, a strong feature-only baseline, gets a respectable 82.6% accuracy. But look at its recall: 67.3%. It misses one in three criminal wallets.

A basic graph network — GCN, no attention — actually does worse than XGBoost. So just adding a graph naively isn't enough.

Our GATv2 with attention gets 88.4% recall — 21 points above XGBoost.

And here's the interesting part: the criminals XGBoost misses are exactly the ones hiding behind normal-looking features — balanced amounts, ordinary timing, nothing suspicious on paper. What they can't fake is who they transact with. Their neighborhood exposes them. That's what attention learns: which connections matter.

So the answer is yes — graph structure matters, but only when the model can learn which neighbors to listen to."

## Slide 11 — Limitations & Future Work (~50 sec)

"To be honest about the limits.

First, our labels come from existing datasets, and criminal behavior drifts — patterns from a few years ago won't all hold tomorrow.

Second, we only look two hops around a wallet. Long laundering chains, designed specifically to put distance between dirty and clean money, can escape that window.

Third, the web tool works, but we haven't run a formal user study with real investigators yet.

Each limitation points at the next step: temporal GNNs that model how behavior evolves over time; graph sampling to scale beyond two hops; and the most practical one — integration at exchanges, scoring wallets in real time at deposit, exactly where screening decisions are made."

---
**Delivery tips:** pause after "we missed only 550" and after "21 points above XGBoost" — let the numbers land. On slide 10, point at the recall bars, not the chart in general. Total ≈ 520 words ≈ 3.5 min at normal pace; trim the calibration paragraph on slide 9 if running long.
