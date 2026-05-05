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


def fetch_transactions_mempool(wallet_address: str) -> List[Dict]:
    """Fetch confirmed transactions for a wallet from mempool.space.

    Uses /txs/chain (max 25 confirmed per page) instead of /txs (50
    confirmed+mempool) to bound payload size on heavy exchange-style wallets,
    which can otherwise return tens of MB.
    """
    print(f"   Fetching transactions from Mempool.space for {wallet_address}...")
    try:
        r = requests.get(
            f"https://mempool.space/api/address/{wallet_address}/txs/chain",
            headers={'User-Agent': 'Mozilla/5.0'},
            timeout=30,
        )
        if r.status_code == 200:
            txs = r.json()
            print(f"   Found {len(txs)} transactions")
            return txs
        print(f"   API returned status {r.status_code}")
    except Exception as e:
        print(f"   Error fetching transactions: {e}")
    return []


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
