"""Feature construction for wave forecasting.

Everything here operates on a regularly-indexed hourly DataFrame from
:mod:`src.data.ndbc`, and every feature is causal: it uses only information
available at forecast issue time. That property is what the leakage tests in
``tests/test_features.py`` check, and it is the single easiest thing to get
wrong in a time-series pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.processing.wave_power_flux import (
    ENERGY_PERIOD_FACTORS,
    calculate_wave_power_flux,
    group_velocity_deep_water,
    steepness,
)

#: Default lags in hours. Dense at short range where autocorrelation carries the
#: signal, sparser further back where it mostly encodes "which storm are we in".
DEFAULT_LAGS = (1, 2, 3, 6, 9, 12, 18, 24, 36, 48, 72)

#: Rolling-window lengths in hours: roughly quarter-day, day, and three days.
DEFAULT_WINDOWS = (6, 24, 72)


def add_direction_features(df: pd.DataFrame, columns=("MWD", "WDIR")) -> pd.DataFrame:
    """Encode compass directions as sine/cosine pairs.

    A direction in degrees is circular: 359 and 1 are two degrees apart but look
    358 apart to a model. Splitting into sin/cos makes the wraparound continuous.
    """
    out = df.copy()
    for column in columns:
        if column in out.columns:
            radians = np.deg2rad(out[column])
            out[f"{column}_sin"] = np.sin(radians)
            out[f"{column}_cos"] = np.cos(radians)
            out = out.drop(columns=[column])
    return out


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived wave quantities: power flux, group velocity, steepness.

    These are deterministic functions of columns the model already has, so they
    add no information in the strict sense. They help anyway, because they
    encode the nonlinear combinations (``H^2 * T``) that the physics actually
    depends on and that a tree would otherwise have to approximate with splits.
    """
    out = df.copy()
    if "WVHT" not in out.columns:
        return out

    # APD is the zero-crossing period; convert to the energy period the flux
    # formula is defined on.
    if "APD" in out.columns:
        energy_period = out["APD"] * ENERGY_PERIOD_FACTORS["VTM02"]
        out["power_flux"] = calculate_wave_power_flux(out["WVHT"], energy_period)
        out["steepness"] = steepness(out["WVHT"], out["APD"])
    if "DPD" in out.columns:
        out["group_velocity"] = group_velocity_deep_water(out["DPD"])
        # Ratio of peak to mean period separates clean long swell from
        # locally-generated wind sea, which behave very differently.
        if "APD" in out.columns:
            out["period_ratio"] = out["DPD"] / out["APD"].replace(0, np.nan)
    return out


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Seasonal and diurnal harmonics.

    Pacific Northwest wave energy has a strong annual cycle - winter storms
    dominate. Encoding day-of-year as a sin/cos pair lets a linear model
    represent that cycle with two coefficients instead of a month dummy set,
    and keeps December adjacent to January.
    """
    out = df.copy()
    day_of_year = out.index.dayofyear.to_numpy()
    hour = out.index.hour.to_numpy()

    out["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    # Second harmonic captures the asymmetry between a long calm summer and a
    # short intense winter.
    out["doy_sin2"] = np.sin(4 * np.pi * day_of_year / 365.25)
    out["doy_cos2"] = np.cos(4 * np.pi * day_of_year / 365.25)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return out


def add_lag_features(
    df: pd.DataFrame,
    columns: list[str],
    lags=DEFAULT_LAGS,
) -> pd.DataFrame:
    """Past values of the given columns.

    Uses ``shift`` on a regular hourly index, so lag N is exactly N hours back.
    On an irregular index it would be N *observations* back, which is a
    different and usually wrong thing.
    """
    out = df.copy()
    for column in columns:
        if column not in df.columns:
            continue
        for lag in lags:
            out[f"{column}_lag{lag}"] = df[column].shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    columns: list[str],
    windows=DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Trailing mean, standard deviation, and trend over each window.

    ``closed="left"`` excludes the current timestamp, so the window covers only
    strictly past observations. Without it the feature at time t includes the
    observation at time t, which for a nowcast is harmless but for evaluation
    against a forecast issued at t is a subtle leak.
    """
    out = df.copy()
    for column in columns:
        if column not in df.columns:
            continue
        series = df[column]
        for window in windows:
            roll = series.rolling(f"{window}h", closed="left")
            out[f"{column}_mean{window}"] = roll.mean()
            out[f"{column}_std{window}"] = roll.std()
            # Change over the window: is the sea building or dying?
            out[f"{column}_delta{window}"] = series.shift(1) - series.shift(window)
    return out


def build_feature_frame(
    df: pd.DataFrame,
    lag_columns: list[str] | None = None,
    lags=DEFAULT_LAGS,
    windows=DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Full causal feature set for one station.

    Args:
        df: Hourly observations from :func:`src.data.ndbc.load_station`.
        lag_columns: Columns to lag and roll. Defaults to the wave and wind
            variables plus derived power flux.
        lags: Lags in hours.
        windows: Rolling window lengths in hours.

    Returns:
        Feature DataFrame on the same index. Early rows contain NaN where the
        longest lag reaches before the record starts; drop them per-horizon
        rather than globally so short-horizon models keep more data.
    """
    out = add_physics_features(df)
    out = add_direction_features(out)

    if lag_columns is None:
        lag_columns = [
            c
            for c in ("WVHT", "APD", "DPD", "WSPD", "power_flux", "MWD_sin", "MWD_cos")
            if c in out.columns
        ]

    out = add_lag_features(out, lag_columns, lags)
    out = add_rolling_features(out, [c for c in ("WVHT", "WSPD") if c in out.columns], windows)
    out = add_calendar_features(out)
    return out


def make_supervised(
    features: pd.DataFrame,
    target: pd.Series,
    horizon: int,
    feature_columns: list[str] | None = None,
    required_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Align features at issue time t with the target at t + horizon.

    Rows are dropped only when the target or a *required* feature is missing -
    never for a NaN anywhere in the feature set. That distinction matters more
    than it looks. A real buoy record is ~13% missing, and with ~77 lag
    features a row needs every one of them present to survive a blanket
    ``dropna``: P = 0.87^77, roughly one row in 65,000. An earlier version did
    exactly that and silently produced an empty training set on real data while
    working fine on cleaner synthetic input.

    Remaining NaNs are left in place. LightGBM handles them natively; models
    that cannot (ridge) impute inside their own pipeline, fitted on training
    data only.

    Args:
        features: Causal features indexed by issue time.
        target: Observed series to predict.
        horizon: Lead time in hours.
        feature_columns: Subset of feature columns to use. Defaults to all.
        required_columns: Columns that must be present for a row to be usable.
            Defaults to the target variable at issue time, which persistence
            needs to produce a forecast at all - without it there is no
            reference to score against.

    Returns:
        (X, y), aligned, with the target complete and required features
        present. Other features may contain NaN.
    """
    y = target.shift(-horizon)
    y.name = f"{target.name}_h{horizon}"

    X = features if feature_columns is None else features[feature_columns]

    if required_columns is None:
        required_columns = [str(target.name)] if target.name in X.columns else []

    combined = X.join(y, how="inner")
    combined = combined.dropna(subset=[y.name, *required_columns])
    return combined.drop(columns=[y.name]), combined[y.name]
