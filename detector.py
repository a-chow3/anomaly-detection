#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Optional

## Get shared logger ##
import logging
logger = logging.getLogger(__name__)

class AnomalyDetector:
    def __init__(self, z_threshold: float = 3.0, contamination: float = 0.05):
        self.z_threshold = z_threshold
        self.contamination = contamination

    def zscore_flag(self, values: pd.Series, mean: float, std: float) -> pd.Series:
        """
        Flag values more than z_threshold standard deviations from the
        established baseline mean. Returns a Series of z-scores.
        """
        ## Guard against zero std to avoid division errors ##
        if std == 0:
            logger.warning("std is 0 for z-score calculation — returning zeros.")
            return pd.Series([0.0] * len(values))
        return (values - mean).abs() / std

    def isolation_forest_flag(self, df: pd.DataFrame, numeric_cols: list[str]) -> np.ndarray:
        """
        Multivariate anomaly detection across all numeric channels simultaneously.
        IsolationForest returns -1 for anomalies, 1 for normal points.
        """
        ## Wrap IsolationForest fit/predict in try/except ##
        try:
            model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
            X = df[numeric_cols].fillna(df[numeric_cols].median())
            model.fit(X)
            labels = model.predict(X)
            scores = model.decision_function(X)
            logger.info("IsolationForest complete — %d anomalies flagged out of %d rows", (labels == -1).sum(), len(df))
            return labels, scores
        except Exception as e:
            logger.error("IsolationForest failed: %s", e)
            raise

    def run(self, df: pd.DataFrame, numeric_cols: list[str], baseline: dict, method: str = "both") -> pd.DataFrame:
        ## Wrap full detection run in try/except ##
        try:
            result = df.copy()

            # --- Z-score per channel ---
            if method in ("zscore", "both"):
                for col in numeric_cols:
                    ## Wrap per-channel z-score in try/except so one bad column doesn't abort all ##
                    try:
                        stats = baseline.get(col)
                        if stats and stats["count"] >= 30:
                            z_scores = self.zscore_flag(df[col], stats["mean"], stats["std"])
                            result[f"{col}_zscore"] = z_scores.round(4)
                            result[f"{col}_zscore_flag"] = z_scores > self.z_threshold
                            logger.info("Z-score computed for channel %s (baseline count=%d)", col, stats["count"])
                        else:
                            ## Log when baseline is immature ##
                            count = stats["count"] if stats else 0
                            logger.warning("Skipping z-score for %s — insufficient baseline (count=%d, need 30)", col, count)
                            result[f"{col}_zscore"] = None
                            result[f"{col}_zscore_flag"] = None
                    except Exception as e:
                        logger.error("Z-score failed for column %s: %s", col, e)
                        result[f"{col}_zscore"] = None
                        result[f"{col}_zscore_flag"] = None

            # --- IsolationForest across all channels ---
            if method in ("isolation", "both"):
                labels, scores = self.isolation_forest_flag(df, numeric_cols)
                result["if_label"] = labels
                result["if_score"] = scores.round(4)
                result["if_flag"] = labels == -1

            # --- Consensus flag ---
            if method == "both":
                zscore_flags = [
                    result[f"{col}_zscore_flag"]
                    for col in numeric_cols
                    if f"{col}_zscore_flag" in result.columns
                    and result[f"{col}_zscore_flag"].notna().any()
                ]
                if zscore_flags:
                    any_zscore = pd.concat(zscore_flags, axis=1).any(axis=1)
                    result["anomaly"] = any_zscore | result["if_flag"]
                else:
                    result["anomaly"] = result["if_flag"]

            return result

        except Exception as e:
            logger.error("AnomalyDetector.run failed: %s", e)
            raise
