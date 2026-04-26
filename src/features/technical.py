"""
Technical indicator feature engineering module.
Computes derived features from raw OHLCV data.

Feature engineering choices:
  - Momentum indicators (RSI, MACD) capture trend direction
  - Volatility indicators (Bollinger Bands, ATR) capture risk environment
  - Volume indicators (OBV, Volume MA ratio) capture market participation
  - Price ratios are normalized (log scale) to be stationary

AI-generated with Claude Code; reviewed and adapted by student.
"""

import numpy as np
import pandas as pd
from typing import List


class TechnicalIndicators:
    """
    Computes a comprehensive set of technical indicators from OHLCV data.
    All methods are stateless and operate on pandas DataFrames.
    """

    # ------------------------------------------------------------------
    # Momentum
    # ------------------------------------------------------------------

    @staticmethod
    def rsi(close: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index (0-100). Values below 30 = oversold, above 70 = overbought."""
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(span=window, adjust=False).mean()
        avg_loss = loss.ewm(span=window, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """MACD line, signal line, and histogram."""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return pd.DataFrame({
            "MACD": macd_line,
            "MACD_Signal": signal_line,
            "MACD_Hist": histogram,
        })

    @staticmethod
    def rate_of_change(close: pd.Series, window: int = 10) -> pd.Series:
        """Price Rate of Change: percentage change over window."""
        return close.pct_change(periods=window) * 100

    # ------------------------------------------------------------------
    # Volatility
    # ------------------------------------------------------------------

    @staticmethod
    def bollinger_bands(
        close: pd.Series, window: int = 20, n_std: float = 2.0
    ) -> pd.DataFrame:
        """Bollinger Bands: %B position and bandwidth."""
        sma = close.rolling(window).mean()
        std = close.rolling(window).std()
        upper = sma + n_std * std
        lower = sma - n_std * std
        pct_b = (close - lower) / (upper - lower + 1e-10)
        bandwidth = (upper - lower) / (sma + 1e-10)
        return pd.DataFrame({
            "BB_PctB": pct_b,
            "BB_Bandwidth": bandwidth,
        })

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range: measures volatility."""
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(span=window, adjust=False).mean()

    @staticmethod
    def historical_volatility(close: pd.Series, window: int = 20) -> pd.Series:
        """Rolling realized volatility of log returns."""
        log_ret = np.log(close / close.shift(1))
        return log_ret.rolling(window).std() * np.sqrt(252)

    # ------------------------------------------------------------------
    # Trend
    # ------------------------------------------------------------------

    @staticmethod
    def sma_ratio(close: pd.Series, short: int = 10, long: int = 50) -> pd.Series:
        """Ratio of short SMA to long SMA (> 1 = uptrend)."""
        return close.rolling(short).mean() / (close.rolling(long).mean() + 1e-10)

    @staticmethod
    def ema_ratio(close: pd.Series, short: int = 12, long: int = 26) -> pd.Series:
        """Ratio of short EMA to long EMA."""
        ema_s = close.ewm(span=short, adjust=False).mean()
        ema_l = close.ewm(span=long, adjust=False).mean()
        return ema_s / (ema_l + 1e-10)

    # ------------------------------------------------------------------
    # Volume
    # ------------------------------------------------------------------

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume: cumulative volume signed by price direction."""
        direction = np.sign(close.diff())
        return (direction * volume).cumsum()

    @staticmethod
    def volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
        """Current volume relative to rolling mean (surge detection)."""
        return volume / (volume.rolling(window).mean() + 1e-10)

    # ------------------------------------------------------------------
    # Composite builder
    # ------------------------------------------------------------------

    @classmethod
    def compute_all(cls, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Computes all technical indicators and returns a feature DataFrame.
        Input must have columns: Open, High, Low, Close, Volume.
        """
        close = ohlcv["Close"]
        high = ohlcv["High"]
        low = ohlcv["Low"]
        volume = ohlcv["Volume"]

        features = pd.DataFrame(index=ohlcv.index)

        # Price-based log return features
        features["LogReturn_1d"] = np.log(close / close.shift(1))
        features["LogReturn_5d"] = np.log(close / close.shift(5))
        features["LogReturn_20d"] = np.log(close / close.shift(20))

        # Price position features
        features["HighLow_Range"] = (high - low) / (close + 1e-10)
        features["CloseOpen_Gap"] = np.log(close / ohlcv["Open"])

        # Momentum
        features["RSI_14"] = cls.rsi(close, 14)
        features["RSI_7"] = cls.rsi(close, 7)
        macd_df = cls.macd(close)
        features = pd.concat([features, macd_df], axis=1)
        features["ROC_10"] = cls.rate_of_change(close, 10)
        features["ROC_5"] = cls.rate_of_change(close, 5)

        # Volatility
        bb_df = cls.bollinger_bands(close)
        features = pd.concat([features, bb_df], axis=1)
        features["ATR_14"] = cls.atr(high, low, close, 14)
        features["HV_20"] = cls.historical_volatility(close, 20)

        # Trend
        features["SMA_Ratio_10_50"] = cls.sma_ratio(close, 10, 50)
        features["EMA_Ratio_12_26"] = cls.ema_ratio(close, 12, 26)

        # Volume
        obv_raw = cls.obv(close, volume)
        # Normalize OBV by its rolling std to make it comparable across stocks
        features["OBV_Norm"] = obv_raw / (obv_raw.rolling(50).std() + 1e-10)
        features["Volume_Ratio"] = cls.volume_ratio(volume, 20)

        return features

    @classmethod
    def feature_names(cls) -> List[str]:
        """Returns expected feature column names (useful for ablation studies)."""
        return [
            "LogReturn_1d", "LogReturn_5d", "LogReturn_20d",
            "HighLow_Range", "CloseOpen_Gap",
            "RSI_14", "RSI_7", "MACD", "MACD_Signal", "MACD_Hist",
            "ROC_10", "ROC_5",
            "BB_PctB", "BB_Bandwidth", "ATR_14", "HV_20",
            "SMA_Ratio_10_50", "EMA_Ratio_12_26",
            "OBV_Norm", "Volume_Ratio",
        ]

    @classmethod
    def momentum_features(cls) -> List[str]:
        return ["RSI_14", "RSI_7", "MACD", "MACD_Signal", "MACD_Hist", "ROC_10", "ROC_5"]

    @classmethod
    def volatility_features(cls) -> List[str]:
        return ["BB_PctB", "BB_Bandwidth", "ATR_14", "HV_20"]

    @classmethod
    def trend_features(cls) -> List[str]:
        return ["SMA_Ratio_10_50", "EMA_Ratio_12_26", "LogReturn_5d", "LogReturn_20d"]
