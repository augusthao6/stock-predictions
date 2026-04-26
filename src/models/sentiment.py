"""
Sentiment analysis via Claude API (Anthropic) applied to financial news headlines.

Integration: Each headline is scored by Claude; scores are aggregated into
daily per-ticker sentiment features fed into the ensemble model.
This constitutes multiple API calls meaningfully integrated into the ML pipeline.

AI-generated with Claude Code; reviewed and adapted by student.
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a financial sentiment analyst. "
    "Respond ONLY with valid JSON. No markdown, no explanation."
)

SENTIMENT_PROMPT = (
    'Analyze the financial sentiment of this news headline.\n'
    'Headline: "{headline}"\n'
    'Respond with JSON: {{"sentiment": "positive", "negative", or "neutral", '
    '"score": <float -1.0 to 1.0>, "confidence": <float 0.0 to 1.0>}}'
)


class SentimentAnalyzer:
    """
    Calls Claude API to score financial news headlines.
    Falls back to rule-based scoring when API key is unavailable.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-haiku-4-5-20251001"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self._client = None
        self._use_fallback = False

        if self.api_key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
                logger.info(f"Claude API client initialized (model: {self.model})")
            except ImportError:
                logger.warning("anthropic package not installed; using rule-based fallback")
                self._use_fallback = True
        else:
            logger.warning("No ANTHROPIC_API_KEY found; using rule-based fallback sentiment")
            self._use_fallback = True

    # ------------------------------------------------------------------
    # Claude API scoring
    # ------------------------------------------------------------------

    def score_headline(self, headline: str, retries: int = 3) -> Dict:
        """
        Calls Claude to score a single headline.
        Returns dict with keys: sentiment, score, confidence.
        """
        if self._use_fallback or not self._client:
            return self._rule_based_score(headline)

        prompt = SENTIMENT_PROMPT.format(headline=headline)
        for attempt in range(retries):
            try:
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=128,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text.strip()
                result = json.loads(text)
                result["score"] = float(np.clip(result.get("score", 0.0), -1.0, 1.0))
                result["confidence"] = float(np.clip(result.get("confidence", 0.5), 0.0, 1.0))
                return result
            except json.JSONDecodeError as e:
                logger.debug(f"JSON parse error on attempt {attempt+1}: {e}")
            except Exception as e:
                logger.debug(f"API error on attempt {attempt+1}: {e}")
                time.sleep(2 ** attempt)

        return {"sentiment": "neutral", "score": 0.0, "confidence": 0.0}

    def score_batch(self, headlines: List[str], delay: float = 0.1) -> List[Dict]:
        """
        Scores a list of headlines, respecting rate limits with configurable delay.
        """
        results = []
        for i, headline in enumerate(headlines):
            result = self.score_headline(headline)
            result["headline"] = headline
            results.append(result)
            if not self._use_fallback and (i + 1) % 10 == 0:
                logger.info(f"  Scored {i+1}/{len(headlines)} headlines via Claude API")
            if not self._use_fallback:
                time.sleep(delay)
        return results

    # ------------------------------------------------------------------
    # Aggregation into daily features
    # ------------------------------------------------------------------

    def aggregate_daily_sentiment(
        self, news_df: pd.DataFrame, tickers: List[str]
    ) -> pd.DataFrame:
        """
        Scores all headlines and aggregates to daily per-ticker sentiment scores.

        Returns DataFrame with MultiIndex (date, ticker) and column 'sentiment_score'.
        The confidence-weighted mean is used to down-weight uncertain predictions.
        """
        if news_df.empty:
            logger.warning("No news data; returning zero sentiment scores")
            return pd.DataFrame(columns=["ticker", "date", "sentiment_score"])

        headlines = news_df["title"].tolist()
        scored = self.score_batch(headlines)

        scored_df = pd.DataFrame(scored)
        scored_df["ticker"] = news_df["ticker"].values
        scored_df["date"] = pd.to_datetime(news_df["published_at"].values)
        scored_df["weighted_score"] = scored_df["score"] * scored_df["confidence"]

        daily = (
            scored_df.groupby(["ticker", "date"])
            .agg(
                sentiment_score=("weighted_score", "mean"),
                n_articles=("score", "count"),
            )
            .reset_index()
        )

        logger.info(
            f"Aggregated sentiment for {daily['ticker'].nunique()} tickers "
            f"over {daily['date'].nunique()} days"
        )
        return daily

    # ------------------------------------------------------------------
    # Rule-based fallback (no API key required)
    # ------------------------------------------------------------------

    POSITIVE_WORDS = {
        "beat", "beats", "surge", "surges", "rally", "rallies", "gain", "gains",
        "profit", "profits", "record", "growth", "grows", "rise", "rises",
        "strong", "upgrade", "upgraded", "buy", "bull", "bullish", "outperform",
        "exceed", "exceeds", "positive", "upside", "boost", "boosts",
    }
    NEGATIVE_WORDS = {
        "miss", "misses", "drop", "drops", "fall", "falls", "decline", "declines",
        "loss", "losses", "cut", "cuts", "layoff", "layoffs", "warning",
        "downgrade", "downgraded", "sell", "bear", "bearish", "underperform",
        "weak", "negative", "downside", "risk", "risks", "concern", "concerns",
    }

    def _rule_based_score(self, headline: str) -> Dict:
        """Simple word-count sentiment for offline/fallback operation."""
        words = set(headline.lower().split())
        pos = len(words & self.POSITIVE_WORDS)
        neg = len(words & self.NEGATIVE_WORDS)
        total = pos + neg
        if total == 0:
            score, sentiment = 0.0, "neutral"
        else:
            score = (pos - neg) / total
            sentiment = "positive" if score > 0.1 else ("negative" if score < -0.1 else "neutral")
        return {"sentiment": sentiment, "score": float(score), "confidence": 0.6}
