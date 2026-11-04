"""
data/quant_models.py — Specialized statistical models for regime detection and volatility.
Contains logic for HMM, GARCH, and ARIMA features.

CRITICAL: All models use ROLLING/EXPANDING windows to prevent look-ahead bias.
"""

import numpy as np
import pandas as pd
import logging
from hmmlearn.hmm import GaussianHMM
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

ROLLING_WINDOW = 500  # Look-back window for all quant models


def compute_hmm_regimes(returns: pd.Series, n_components: int = 3) -> pd.Series:
    """
    Fits a Gaussian HMM using a ROLLING window to prevent look-ahead bias.
    At each point t, only data [t-ROLLING_WINDOW : t] is used.
    For efficiency, we refit every 50 candles instead of every single one.
    """
    data = returns.dropna()
    if len(data) < ROLLING_WINDOW:
        return pd.Series(index=returns.index, data=0.0)

    result = pd.Series(index=data.index, data=0.0, dtype=float)
    refit_interval = 50

    last_model = None
    for i in range(ROLLING_WINDOW, len(data)):
        if i % refit_interval == 0 or last_model is None:
            window = data.iloc[max(0, i - ROLLING_WINDOW):i].values.reshape(-1, 1)
            try:
                model = GaussianHMM(
                    n_components=n_components,
                    covariance_type="full",
                    n_iter=50,
                    random_state=42
                )
                model.fit(window)
                last_model = model
            except Exception:
                pass

        if last_model is not None:
            try:
                point = data.iloc[i:i+1].values.reshape(-1, 1)
                state = last_model.predict(point)[0]
                result.iloc[i] = float(state)
            except Exception:
                result.iloc[i] = 0.0

    return result.reindex(returns.index).ffill().fillna(0.0)


def compute_garch_volatility(returns: pd.Series) -> pd.Series:
    """
    Fits a GARCH(1,1) model using an EXPANDING window to prevent look-ahead bias.
    Refits every 100 candles for efficiency.
    """
    data = returns.dropna()
    if len(data) < ROLLING_WINDOW:
        return pd.Series(index=returns.index, data=0.0)

    rescaled = data * 100
    result = pd.Series(index=data.index, data=0.0, dtype=float)
    refit_interval = 100

    for i in range(ROLLING_WINDOW, len(rescaled)):
        if i % refit_interval == 0:
            window = rescaled.iloc[max(0, i - ROLLING_WINDOW):i]
            try:
                model = arch_model(window, vol='Garch', p=1, q=1, dist='normal')
                res = model.fit(disp='off', show_warning=False)
                cond_vol = res.conditional_volatility / 100.0
                # Fill forward from the fit window
                end_idx = min(i + refit_interval, len(rescaled))
                for j in range(i, end_idx):
                    if j < len(cond_vol):
                        result.iloc[j] = cond_vol.iloc[-1]
                    else:
                        result.iloc[j] = cond_vol.iloc[-1]
            except Exception:
                pass

    return result.reindex(returns.index).ffill().fillna(0.0)


def compute_arima_forecast(returns: pd.Series, order=(1, 0, 1)) -> pd.Series:
    """
    Fits an ARIMA model using a ROLLING window to prevent look-ahead bias.
    Returns the 1-step ahead forecast error (residual).
    Refits every 100 candles for efficiency.
    """
    data = returns.dropna()
    if len(data) < ROLLING_WINDOW:
        return pd.Series(index=returns.index, data=0.0)

    result = pd.Series(index=data.index, data=0.0, dtype=float)
    refit_interval = 100
    last_resid = 0.0

    for i in range(ROLLING_WINDOW, len(data)):
        if i % refit_interval == 0:
            window = data.iloc[max(0, i - ROLLING_WINDOW):i]
            try:
                model = ARIMA(window, order=order)
                res = model.fit()
                last_resid = res.resid.iloc[-1]
            except Exception:
                last_resid = 0.0

        result.iloc[i] = last_resid

    return result.reindex(returns.index).ffill().fillna(0.0)
