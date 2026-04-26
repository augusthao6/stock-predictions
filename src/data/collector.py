"""
Data collection module: downloads stock price data via yfinance API
and financial news headlines via web requests.

AI-generated with Claude Code; reviewed and adapted by student.
"""

import os
import time
import json
import logging
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM", "BRK-B", "UNH"]

DATA_DIR = Path(__file__).parents[3] / "data"


class StockDataCollector:
    """
    Downloads and caches multi-stock OHLCV data from Yahoo Finance.
    Handles rate limiting and partial failures gracefully.
    """

    def __init__(self, tickers: List[str] = TICKERS, data_dir: Path = DATA_DIR):
        self.tickers = tickers
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_prices(
        self,
        start: str = "2015-01-01",
        end: str = "2024-01-01",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Downloads adjusted OHLCV data for all tickers.
        Returns a MultiIndex DataFrame (date x ticker x field).
        Caches results to avoid redundant API calls.
        """
        cache_path = self.data_dir / f"prices_{start}_{end}.parquet"

        if cache_path.exists() and not force_refresh:
            logger.info(f"Loading cached prices from {cache_path}")
            return pd.read_parquet(cache_path)

        logger.info(f"Downloading price data for {len(self.tickers)} tickers ({start} to {end})")

        frames = {}
        for ticker in self.tickers:
            for attempt in range(3):
                try:
                    df = yf.download(
                        ticker,
                        start=start,
                        end=end,
                        auto_adjust=True,
                        progress=False,
                    )
                    if df.empty:
                        logger.warning(f"No data returned for {ticker}")
                        break

                    # Flatten multi-level columns if present
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)

                    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                    df.index = pd.to_datetime(df.index)
                    df.index.name = "Date"
                    frames[ticker] = df
                    logger.info(f"  {ticker}: {len(df)} rows")
                    break
                except Exception as e:
                    logger.warning(f"  {ticker} attempt {attempt + 1} failed: {e}")
                    time.sleep(2 ** attempt)

        if not frames:
            raise RuntimeError("Failed to download any price data")

        combined = pd.concat(frames, axis=1, keys=frames.keys())
        combined.columns.names = ["Ticker", "Field"]
        combined.to_parquet(cache_path)
        logger.info(f"Saved {len(combined)} rows to {cache_path}")
        return combined

    def fetch_news_headlines(
        self,
        ticker: str,
        n_articles: int = 50,
    ) -> List[Dict]:
        """
        Fetches recent financial news headlines for a ticker using Yahoo Finance's
        built-in news endpoint. Returns list of {title, publisher, published_at} dicts.
        """
        try:
            stock = yf.Ticker(ticker)
            news_items = stock.news or []
            results = []
            for item in news_items[:n_articles]:
                results.append({
                    "ticker": ticker,
                    "title": item.get("title", ""),
                    "publisher": item.get("publisher", ""),
                    "published_at": datetime.fromtimestamp(
                        item.get("providerPublishTime", 0)
                    ).strftime("%Y-%m-%d"),
                })
            return results
        except Exception as e:
            logger.warning(f"Could not fetch news for {ticker}: {e}")
            return []

    def collect_all_news(self, n_articles_per_ticker: int = 30) -> pd.DataFrame:
        """
        Collects news headlines for all tickers and saves to CSV.
        Returns a DataFrame of news items ready for sentiment analysis.
        """
        cache_path = self.data_dir / "news_headlines.csv"
        if cache_path.exists():
            logger.info("Loading cached news headlines")
            return pd.read_csv(cache_path, parse_dates=["published_at"])

        all_news = []
        for ticker in self.tickers:
            items = self.fetch_news_headlines(ticker, n_articles_per_ticker)
            all_news.extend(items)
            time.sleep(0.5)  # be polite to Yahoo Finance

        if not all_news:
            logger.warning("No news collected; returning empty DataFrame")
            return pd.DataFrame(columns=["ticker", "title", "publisher", "published_at"])

        df = pd.DataFrame(all_news)
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
        df = df.dropna(subset=["title"]).drop_duplicates(subset=["title"])
        df.to_csv(cache_path, index=False)
        logger.info(f"Collected {len(df)} news headlines for {len(self.tickers)} tickers")
        return df

    def get_collection_summary(self, prices: pd.DataFrame) -> Dict:
        """Returns metadata about the collected dataset for documentation."""
        tickers = prices.columns.get_level_values(0).unique().tolist()
        return {
            "tickers": tickers,
            "n_tickers": len(tickers),
            "date_range": f"{prices.index.min().date()} to {prices.index.max().date()}",
            "n_trading_days": len(prices),
            "total_observations": len(prices) * len(tickers),
            "fields": ["Open", "High", "Low", "Close", "Volume"],
            "source": "Yahoo Finance API via yfinance",
            "collection_method": "Programmatic API integration with retry logic and caching",
        }
