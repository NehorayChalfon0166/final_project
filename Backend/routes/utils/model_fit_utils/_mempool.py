"""Mempool.space HTTP fetchers with TTL caching and in-flight dedup.

Both transaction history and address stats follow the same pattern: check
the cache, and if a fetch for the same address is already in flight, wait on
its event instead of issuing a duplicate HTTP call (which would otherwise hit
mempool.space's same-IP throttling).
"""
import threading
import time
from typing import Any, Callable, Dict, List

import requests

from . import _cache


def _mempool_get(url: str, max_retries: int = 3, backoff: float = 5.0, timeout: int = 15):
    """GET with backoff on 429 / transient errors. Mirrors notebooks/colab_wallet_analyzer.ipynb _get."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=timeout, headers=headers)
            if r.status_code == 429:
                time.sleep(backoff)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(backoff)
    raise RuntimeError(f"mempool.space request failed after {max_retries} tries: {last_err}")


def fetch_transactions_mempool(
    wallet_address: str,
    max_pages: int = 2,
    sleep_between: float = 0.5,
) -> List[Dict]:
    """Fetch transactions for a wallet from mempool.space.

    Matches notebooks/colab_wallet_analyzer.ipynb: first page from /txs
    (~50 txs, confirmed+mempool) then paginate via /txs/chain/{last_txid}
    (~25 confirmed each). Cap at max_pages pages so heavy exchange wallets
    don't blow up payload size. The graph builder filters unconfirmed txs.
    """
    print(f"   Fetching transactions from Mempool.space for {wallet_address}...")
    try:
        first = _mempool_get(f"https://mempool.space/api/address/{wallet_address}/txs")
    except RuntimeError as e:
        print(f"   {e}")
        return []
    if not first:
        print("   Found 0 transactions")
        return []

    all_txs = list(first)
    last_id = first[-1].get('txid')
    for _ in range(max_pages - 1):
        if not last_id:
            break
        time.sleep(sleep_between)
        try:
            page = _mempool_get(
                f"https://mempool.space/api/address/{wallet_address}/txs/chain/{last_id}"
            )
        except RuntimeError:
            break
        if not page:
            break
        all_txs.extend(page)
        last_id = page[-1].get('txid')
        if len(page) < 25:
            break

    print(f"   Found {len(all_txs)} transactions")
    return all_txs


def _fetch_address_stats(wallet_address: str) -> Dict[str, Any]:
    try:
        r = requests.get(
            f"https://mempool.space/api/address/{wallet_address}",
            headers={'User-Agent': 'Mozilla/5.0'},
            timeout=15,
        )
        return r.json() if r.status_code == 200 else {}
    except Exception as e:
        print(f"   [stats-cache] fetch error: {e}")
        return {}


def _cached_fetch(
    wallet_address: str,
    *,
    label: str,
    lock: threading.Lock,
    store: Dict[str, Any],
    inflight: Dict[str, threading.Event],
    fetcher: Callable[[str], Any],
    empty: Any,
) -> Any:
    """Generic TTL+inflight wrapper used by the public cached fetchers."""
    cached = _cache.get_fresh(lock, store, wallet_address)
    if cached is not None:
        print(f"   [{label}-cache] HIT {wallet_address}")
        return cached

    with lock:
        cached = store.get(wallet_address)
        if cached and (time.time() - cached[0]) < _cache.WALLET_TTL_SECONDS:
            return cached[1]
        event = inflight.get(wallet_address)
        if event is None:
            event = threading.Event()
            inflight[wallet_address] = event
            owner = True
        else:
            owner = False

    if not owner:
        event.wait(timeout=_cache.WALLET_TTL_SECONDS)
        cached = _cache.get_fresh(lock, store, wallet_address)
        return cached if cached is not None else empty

    try:
        payload = fetcher(wallet_address)
        with lock:
            store[wallet_address] = (time.time(), payload)
        return payload
    finally:
        with lock:
            inflight.pop(wallet_address, None)
        event.set()


def get_cached_transactions(wallet_address: str) -> List[Dict]:
    """Fetch transactions, deduped via per-address TTL cache."""
    return _cached_fetch(
        wallet_address,
        label='tx',
        lock=_cache.TX_LOCK,
        store=_cache.tx_store(),
        inflight=_cache.tx_inflight(),
        fetcher=fetch_transactions_mempool,
        empty=[],
    )


def get_cached_address_stats(wallet_address: str) -> Dict[str, Any]:
    """Fetch /address stats, deduped via per-address TTL cache."""
    return _cached_fetch(
        wallet_address,
        label='stats',
        lock=_cache.STATS_LOCK,
        store=_cache.stats_store(),
        inflight=_cache.stats_inflight(),
        fetcher=_fetch_address_stats,
        empty={},
    )
