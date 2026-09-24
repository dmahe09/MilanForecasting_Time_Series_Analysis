"""
Three deliberately different one-step-ahead forecasting models:

1. SARIMA               — classical statistical model, captures daily seasonality directly.
2. LightGBM (lag features) — gradient-boosted trees on engineered lag/calendar features.
3. LSTM                 — small recurrent neural network learning temporal dependencies
                          directly from the raw sequence.

Each model exposes: fit(train_series) -> None, predict(history) -> np.ndarray of
one-step-ahead forecasts over the evaluation window, and reports its own
train/predict wall-clock time.
"""

import time
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    lgb = None
    LGB_AVAILABLE = False
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor

import warnings
warnings.filterwarnings("ignore")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM as KerasLSTM, Dense
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


SEASONAL_PERIOD = 144  # 10-min intervals per day


# ---------------------------------------------------------------------------
# 1. SARIMA — implemented as dynamic harmonic regression (ARIMA errors +
#    Fourier terms for daily/weekly seasonality) rather than native
#    SARIMAX(P,D,Q,144). A literal seasonal_order with s=144 forces
#    statsmodels to build state-space matrices sized to the seasonal
#    period, which becomes computationally impractical at 10-minute
#    resolution (fits can take many minutes to hours). Representing
#    seasonality with a handful of Fourier sin/cos regressors instead
#    of a full seasonal ARIMA polynomial is a standard, much cheaper way
#    to handle high-frequency seasonal series (Hyndman & Athanasopoulos,
#    "Forecasting: Principles and Practice", ch. on dynamic regression),
#    and is worth citing as such in the report's Methodology section.
# ---------------------------------------------------------------------------
def _fourier_terms(n: int, period: int, n_harmonics: int, start: int = 0) -> np.ndarray:
    t = np.arange(start, start + n)
    cols = []
    for k in range(1, n_harmonics + 1):
        cols.append(np.sin(2 * np.pi * k * t / period))
        cols.append(np.cos(2 * np.pi * k * t / period))
    return np.column_stack(cols)


class SarimaModel:
    """ARIMA(p,d,q) with exogenous Fourier terms for daily + weekly seasonality."""

    def __init__(self, order=(2, 0, 1), daily_harmonics=3, weekly_harmonics=2):
        self.order = order
        self.daily_harmonics = daily_harmonics
        self.weekly_harmonics = weekly_harmonics
        self.fitted = None
        self.train_time_s = None
        self.n_train = None

    def _exog(self, n: int, start: int) -> np.ndarray:
        daily = _fourier_terms(n, SEASONAL_PERIOD, self.daily_harmonics, start)
        weekly = _fourier_terms(n, SEASONAL_PERIOD * 7, self.weekly_harmonics, start)
        return np.hstack([daily, weekly])

    def fit(self, train_series: pd.Series):
        t0 = time.time()
        self.n_train = len(train_series)
        exog = self._exog(self.n_train, start=0)
        model = SARIMAX(
            train_series.values, exog=exog, order=self.order,
            enforce_stationarity=False, enforce_invertibility=False,
        )
        self.fitted = model.fit(disp=False)
        self.train_time_s = time.time() - t0

    def predict_n_steps(self, n_steps: int) -> np.ndarray:
        t0 = time.time()
        future_exog = self._exog(n_steps, start=self.n_train)
        forecast = self.fitted.forecast(steps=n_steps, exog=future_exog)
        self.predict_time_s = time.time() - t0
        return np.asarray(forecast)


# ---------------------------------------------------------------------------
# 2. LightGBM on lag + calendar features
# ---------------------------------------------------------------------------
def make_lag_features(series: pd.Series, n_lags: int = 12, extra_lags=(144, 1008)):
    """
    Builds a supervised-learning table: recent lags + daily/weekly lag +
    hour-of-day / day-of-week calendar features -> next-step target.
    """
    df = pd.DataFrame({"y": series})
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    for lag in extra_lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek
    df["target"] = df["y"].shift(-1)
    df = df.drop(columns=["y"]).dropna()
    return df


class LightGBMModel:
    def __init__(self, n_lags=12, extra_lags=(144, 1008), params=None):
        self.n_lags = n_lags
        self.extra_lags = extra_lags
        self.params = params or dict(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            max_depth=-1, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=-1,
        )
        if LGB_AVAILABLE:
            self.model = lgb.LGBMRegressor(**self.params)
        else:
            gb_params = {k: v for k, v in self.params.items() if k in (
                'n_estimators', 'learning_rate', 'max_depth', 'random_state'
            )}
            # Ensure sklearn receives a valid max_depth (None or positive int).
            # Map LightGBM's -1 convention to None and coerce other non-positive
            # values to None as well.
            if 'max_depth' in gb_params:
                try:
                    if int(gb_params['max_depth']) <= 0:
                        gb_params['max_depth'] = None
                    else:
                        gb_params['max_depth'] = int(gb_params['max_depth'])
                except Exception:
                    gb_params['max_depth'] = None
            # Fallback to sklearn's GradientBoostingRegressor when LightGBM is
            # not available (e.g., missing native libomp on macOS).
            self.model = GradientBoostingRegressor(**gb_params)
        self.train_time_s = None
        self.feature_cols = None

    def fit(self, train_series: pd.Series):
        feat = make_lag_features(train_series, self.n_lags, self.extra_lags)
        X, y = feat.drop(columns=["target"]), feat["target"]
        self.feature_cols = X.columns.tolist()
        t0 = time.time()
        self.model.fit(X, y)
        self.train_time_s = time.time() - t0

    def predict_recursive(self, history: pd.Series, n_steps: int) -> np.ndarray:
        """Recursive one-step-ahead forecasting over the evaluation window."""
        t0 = time.time()
        extended = history.copy()
        preds = []
        for _ in range(n_steps):
            feat_row = make_lag_features(extended, self.n_lags, self.extra_lags).iloc[[-1]]
            # last row has NaN target (expected) since it's the point we predict
            feat_row = feat_row[self.feature_cols]
            pred = self.model.predict(feat_row)[0]
            preds.append(pred)
            next_ts = extended.index[-1] + (extended.index[1] - extended.index[0])
            extended.loc[next_ts] = pred
        self.predict_time_s = time.time() - t0
        return np.array(preds)


# ---------------------------------------------------------------------------
# 3. Small single-layer LSTM
# ---------------------------------------------------------------------------
class LstmModel:
    def __init__(self, seq_len=72, units=32, epochs=30, batch_size=256):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available in this environment.")
        self.seq_len = seq_len
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler()
        self.model = None
        self.train_time_s = None

    def _make_sequences(self, values: np.ndarray):
        X, y = [], []
        for i in range(len(values) - self.seq_len):
            X.append(values[i:i + self.seq_len])
            y.append(values[i + self.seq_len])
        return np.array(X), np.array(y)

    def fit(self, train_series: pd.Series):
        values = self.scaler.fit_transform(train_series.values.reshape(-1, 1)).flatten()
        X, y = self._make_sequences(values)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        self.model = Sequential([
            KerasLSTM(self.units, input_shape=(self.seq_len, 1)),
            Dense(1),
        ])
        self.model.compile(optimizer="adam", loss="mse")

        t0 = time.time()
        self.model.fit(
            X, y, epochs=self.epochs, batch_size=self.batch_size,
            validation_split=0.1, verbose=0,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        )
        self.train_time_s = time.time() - t0

    def predict_recursive(self, history: pd.Series, n_steps: int) -> np.ndarray:
        t0 = time.time()
        scaled_hist = self.scaler.transform(history.values.reshape(-1, 1)).flatten()
        window = list(scaled_hist[-self.seq_len:])
        preds_scaled = []
        for _ in range(n_steps):
            x = np.array(window[-self.seq_len:]).reshape(1, self.seq_len, 1)
            pred = self.model.predict(x, verbose=0)[0, 0]
            preds_scaled.append(pred)
            window.append(pred)
        self.predict_time_s = time.time() - t0
        preds = self.scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
        return preds
