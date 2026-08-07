from typing import List
import pandas as pd
import numpy as np


FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "lag_24",
    "rolling_mean_6",
    "rolling_std_6",
    "rolling_mean_24",
    "rolling_std_24",
    "diff_1",
    "pct_change_1",
]


def make_features(df: pd.DataFrame, target_col: str = "value") -> pd.DataFrame:
    """
    Membuat fitur time series umum.

    Input minimal:
    - timestamp
    - series_id
    - value

    Output:
    - DataFrame dengan fitur lag, rolling, calendar, dan perubahan harga.
    """

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["series_id", "timestamp"])

    # Calendar features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Group berdasarkan series_id agar bisa multi-series
    g = df.groupby("series_id")[target_col]

    # Lag features
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"lag_{lag}"] = g.shift(lag)

    # Rolling features
    # shift(1) penting agar tidak bocor memakai data masa depan
    for window in [6, 24]:
        shifted = g.shift(1)

        df[f"rolling_mean_{window}"] = shifted.groupby(df["series_id"]).transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )

        df[f"rolling_std_{window}"] = shifted.groupby(df["series_id"]).transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )

    # Difference dan percent change
    df["diff_1"] = g.diff(1).groupby(df["series_id"]).shift(1)
    df["pct_change_1"] = g.pct_change().groupby(df["series_id"]).shift(1)

    # Isi missing value sederhana untuk MVP
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df