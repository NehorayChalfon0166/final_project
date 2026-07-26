# Bitcoin Data API Requirements

What our backend needs to fetch from a Bitcoin blockchain data API to run the GNN
wallet-risk classifier. Currently we use the free `https://mempool.space/api`,
which rate-limits same-IP repeats hard (first call ~1.3s, second ~150s on the
same address). We're evaluating alternatives.

---

## Network

- **Bitcoin mainnet only.** No testnet, no other chains.

## Address types we must support

- P2PKH / P2SH (Base58, starts with `1` or `3`)
- Bech32 SegWit (starts with `bc1`)

---

## Endpoints / data we need

### 1. Address summary stats (per wallet)

Equivalent of `GET /address/{address}` on mempool.space. We need:

- `chain_stats.funded_txo_sum` (total sats received, confirmed)
- `chain_stats.spent_txo_sum` (total sats spent, confirmed)
- `chain_stats.funded_txo_count`, `chain_stats.spent_txo_count`
- `chain_stats.tx_count`
- Same four fields under `mempool_stats` (unconfirmed)

Used by `/info` to display balance, totals, tx count.

### 2. Confirmed transaction history (per wallet) — **the heavy one**

Equivalent of `GET /address/{address}/txs/chain` on mempool.space. For each tx we need:

- `status.confirmed` (bool)
- `status.block_time` (unix timestamp)
- `fee` (sats)
- `vin[]` with `prevout.scriptpubkey_address` and `prevout.value`
  - **Critical:** we need the *previous output's address+value already joined in*,
    so we don't have to make a second call per input to fetch the funding tx.
    Mempool gives this; many APIs don't.
- `vout[]` with `scriptpubkey_address` and `value`

Volume:
- Typical wallet: tens to a few hundred txs.
- Heavy wallets (exchanges): tens of thousands. We currently cap at the first
  page (25 confirmed txs) to bound payload size. An API that paginates cleanly
  and lets us cap by count is preferable to one that streams huge JSON blobs.

### 3. Latest block + its txs (for the "random wallet" demo button)

- `GET /blocks/tip/height` → tip height
- `GET /block-height/{h}` → block hash
- `GET /block/{hash}/txs` → list of txs with `vout[].scriptpubkey_address`

We just pick a random output address from the latest block. Low frequency.

---

## Traffic profile

- Interactive single-user demo, not a production service.
- 5-minute in-process TTL cache and in-flight dedup, so repeats inside a session don't re-hit the API.

### NEW: 2-hop ego graph (neighbor nodes get real features too)

We are upgrading from a 1-hop ego graph (center has real features, neighbors are
ghost nodes with zero features) to a **2-hop graph where neighbor nodes also
carry real data**. Practically this means: for the queried wallet, we fan out
to every direct counterparty and **repeat the same two API calls (address stats
+ tx history) for each neighbor** so we can compute their features and discover
their counterparties (the 2-hop layer).

Cost per analyzed wallet, in API calls:

- 1 stats + 1 tx-history call for the center wallet.
- For each unique neighbor of the center: 1 stats + 1 tx-history call.
  - On a typical wallet that's ~20–200 neighbors → **~40–400 calls per analysis**.
  - On heavy wallets we'll cap neighbor count (e.g. top-K by volume) to bound it.

This is the dimension that makes API choice actually matter. The previous
free-tier estimate ("a few hundred lookups/day") was for the 1-hop design and
is now off by 1–2 orders of magnitude. **Plan for thousands of address lookups
per analysis session, not hundreds per day.** Realistic target: support ~50
analyses/day during development → on the order of **5k–20k address calls/day**.

Bursts are also bigger: a single analysis fans out a few hundred calls in
parallel, so per-second concurrency matters, not just daily totals.

---

## What we want from a provider

Rank the candidates on these, with prices:

1. **Free or cheap tier sized for thousands of address calls/day**, given the
   2-hop fan-out described above. Hard caps under ~5k/day are likely too tight.
2. **High burst concurrency** — a single analysis fires hundreds of address
   lookups in parallel. Providers that rate-limit to e.g. 3 req/s will serialize
   each analysis into minutes of wall time.
3. **No aggressive same-IP throttling** on repeated calls to the same address (the mempool.space pain point).
4. **`vin.prevout` pre-joined** in the tx response (saves N extra calls per tx).
5. **Pagination** on address tx history with a count/limit param (we'll cap neighbor histories tightly).
6. **Stable JSON schema** close to the Esplora/mempool.space shape so the migration is just a base-URL swap, ideally.
7. **Latency** under ~2s for an address with a few hundred txs.
8. **API key OK** if it unlocks the throughput we need; no enterprise sales process.

## Candidates to price out (please research)

- mempool.space (current — free, public; rate limits unclear and harsh in practice)
- blockstream.info Esplora (free, same schema)
- BlockCypher
- Blockchain.com data API
- Blockchair
- QuickNode / GetBlock / NowNodes / Chainstack (provider-style)
- BTC.com API
- Tatum
- any others worth knowing about

For each, please report: pricing tiers, free-tier limits, whether `vin.prevout`
is pre-joined, rate-limit policy, and auth model. Flag anything that *doesn't*
support per-address tx history at all (some only do tx-by-hash).
