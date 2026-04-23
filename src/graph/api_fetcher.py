"""
Async API Fetcher
=================
Fetches transaction data from mempool.space with rate limiting and retry logic.
"""
import asyncio
import aiohttp
import time
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
import logging

from .config import (
    MEMPOOL_API_BASE,
    CONCURRENT_REQUESTS,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    RETRY_DELAY,
    MEMPOOL_RATE_LIMIT
)
from .cache_manager import CacheManager

logger = logging.getLogger(__name__)


@dataclass
class FetchResult:
    """Result of a fetch operation."""
    address: str
    success: bool
    transactions: Optional[List[Dict]] = None
    error: Optional[str] = None
    from_cache: bool = False


class MempoolFetcher:
    """Async fetcher with rate limiting and caching."""

    def __init__(self, cache_manager: CacheManager):
        """
        Initialize the fetcher.

        Args:
            cache_manager: CacheManager instance for caching responses
        """
        self.cache_manager = cache_manager
        self.semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
        self._last_request_time = 0.0
        self._request_interval = 1.0 / MEMPOOL_RATE_LIMIT
        self._rate_lock = asyncio.Lock()

    async def _rate_limit_delay(self):
        """Enforce rate limiting between requests (async-safe)."""
        async with self._rate_lock:
            current_time = time.monotonic()
            elapsed = current_time - self._last_request_time
            if elapsed < self._request_interval:
                await asyncio.sleep(self._request_interval - elapsed)
            self._last_request_time = time.monotonic()

    async def fetch_address_transactions(
        self,
        session: aiohttp.ClientSession,
        address: str
    ) -> FetchResult:
        """
        Fetch transactions for a single address with retry logic.

        Args:
            session: aiohttp session
            address: Bitcoin address to fetch

        Returns:
            FetchResult with transactions or error
        """
        # Check cache first
        cached = self.cache_manager.get_cached_transactions(address)
        if cached is not None:
            return FetchResult(
                address=address,
                success=True,
                transactions=cached,
                from_cache=True
            )

        async with self.semaphore:
            for attempt in range(MAX_RETRIES):
                try:
                    await self._rate_limit_delay()

                    url = f"{MEMPOOL_API_BASE}/address/{address}/txs"

                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as response:
                        if response.status == 200:
                            transactions = await response.json()
                            # Cache the result
                            self.cache_manager.save_transactions(address, transactions)
                            return FetchResult(
                                address=address,
                                success=True,
                                transactions=transactions
                            )
                        elif response.status == 429:
                            # Rate limited - wait and retry
                            wait_time = RETRY_DELAY * (attempt + 1)
                            logger.warning(f"Rate limited, waiting {wait_time}s...")
                            await asyncio.sleep(wait_time)
                        elif response.status == 404:
                            # Address not found - return empty
                            self.cache_manager.save_transactions(address, [])
                            return FetchResult(
                                address=address,
                                success=True,
                                transactions=[]
                            )
                        elif response.status >= 500:
                            # Server error - retry
                            if attempt < MAX_RETRIES - 1:
                                wait_time = RETRY_DELAY * (attempt + 1)
                                logger.warning(f"Server error {response.status} for {address}, retrying in {wait_time}s...")
                                await asyncio.sleep(wait_time)
                            else:
                                return FetchResult(
                                    address=address,
                                    success=False,
                                    error=f"HTTP {response.status} after max retries"
                                )
                        else:
                            # Client error (4xx other than 429) - don't retry
                            return FetchResult(
                                address=address,
                                success=False,
                                error=f"HTTP {response.status}"
                            )

                except asyncio.TimeoutError:
                    if attempt < MAX_RETRIES - 1:
                        logger.warning(f"Timeout for {address}, retrying...")
                        await asyncio.sleep(RETRY_DELAY)
                    else:
                        return FetchResult(
                            address=address,
                            success=False,
                            error="Timeout after max retries"
                        )

                except aiohttp.ClientError as e:
                    if attempt < MAX_RETRIES - 1:
                        logger.warning(f"Client error for {address}: {e}, retrying...")
                        await asyncio.sleep(RETRY_DELAY)
                    else:
                        return FetchResult(
                            address=address,
                            success=False,
                            error=str(e)
                        )

                except Exception as e:
                    return FetchResult(
                        address=address,
                        success=False,
                        error=f"Unexpected error: {str(e)}"
                    )

        return FetchResult(
            address=address,
            success=False,
            error="Max retries exceeded"
        )

    async def fetch_batch(
        self,
        addresses: List[str],
        progress_callback: Optional[Callable[[int, int, FetchResult], None]] = None
    ) -> List[FetchResult]:
        """
        Fetch transactions for addresses sequentially to respect rate limits.

        Args:
            addresses: List of addresses to fetch
            progress_callback: Optional callback(completed, total, result)

        Returns:
            List of FetchResults
        """
        connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)
        headers = {'User-Agent': 'BitcoinWalletAnalyzer/1.0'}

        results = []
        async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
            for i, addr in enumerate(addresses):
                result = await self.fetch_address_transactions(session, addr)
                results.append(result)
                if progress_callback:
                    progress_callback(i + 1, len(addresses), result)

        return results

    def fetch_batch_sync(
        self,
        addresses: List[str],
        progress_callback: Optional[Callable[[int, int, FetchResult], None]] = None
    ) -> List[FetchResult]:
        """
        Synchronous wrapper for fetch_batch.

        Args:
            addresses: List of addresses to fetch
            progress_callback: Optional callback

        Returns:
            List of FetchResults
        """
        return asyncio.run(self.fetch_batch(addresses, progress_callback))
