#!/usr/bin/env python3
import json
import math
import boto3
import logging
import os
from datetime import datetime
from typing import Optional

s3 = boto3.client("s3")

## Get the shared logger configured in app.py ##
logger = logging.getLogger(__name__)

## Path to local log file — must match the path set in app.py ##
LOG_FILE = "/opt/anomaly-detection/app.log"

class BaselineManager:
    """
    Maintains a per-channel running baseline using Welford's online algorithm,
    which computes mean and variance incrementally without storing all past data.
    """
    def __init__(self, bucket: str, baseline_key: str = "state/baseline.json"):
        self.bucket = bucket
        self.baseline_key = baseline_key

    def load(self) -> dict:
        ## Differentiate between missing baseline (expected) and real S3 errors ##
        try:
            response = s3.get_object(Bucket=self.bucket, Key=self.baseline_key)
            baseline = json.loads(response["Body"].read())
            logger.info("Baseline loaded from S3: %s", self.baseline_key)
            return baseline
        except s3.exceptions.NoSuchKey:
            logger.info("No existing baseline found at %s — starting fresh.", self.baseline_key)
            return {}
        except Exception as e:
            logger.error("Unexpected error loading baseline from S3: %s", e)
            raise

    def save(self, baseline: dict):
        ## Wrap baseline save and log sync in try/except ##
        try:
            baseline["last_updated"] = datetime.utcnow().isoformat()
            s3.put_object(
                Bucket=self.bucket,
                Key=self.baseline_key,
                Body=json.dumps(baseline, indent=2),
                ContentType="application/json"
            )
            logger.info("Baseline saved to S3: %s (last_updated=%s)", self.baseline_key, baseline["last_updated"])
        except Exception as e:
            logger.error("Failed to save baseline to S3: %s", e)
            raise

        ## Sync log file to S3 every time baseline is saved ##
        self._sync_log_to_s3()

    def _sync_log_to_s3(self):
        """Upload the local log file to s3://BUCKET/logs/app.log after every baseline save."""
        ## Wrap log sync in its own try/except so a log failure never breaks processing ##
        try:
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, "rb") as f:
                    s3.put_object(
                        Bucket=self.bucket,
                        Key="logs/app.log",
                        Body=f.read(),
                        ContentType="text/plain"
                    )
                logger.info("Log file synced to S3: logs/app.log")
            else:
                logger.warning("Log file not found at %s — skipping S3 sync.", LOG_FILE)
        except Exception as e:
            logger.error("Failed to sync log file to S3: %s", e)

    def update(self, baseline: dict, channel: str, new_values: list[float]) -> dict:
        """
        Welford's online algorithm for numerically stable mean and variance.
        Each channel tracks: count, mean, M2 (sum of squared deviations).
        Variance = M2 / count, std = sqrt(variance).
        """
        ## Wrap Welford update in try/except and log new calculations ##
        try:
            if channel not in baseline:
                baseline[channel] = {"count": 0, "mean": 0.0, "M2": 0.0}
                logger.info("New channel added to baseline: %s", channel)

            state = baseline[channel]
            for value in new_values:
                state["count"] += 1
                delta = value - state["mean"]
                state["mean"] += delta / state["count"]
                delta2 = value - state["mean"]
                state["M2"] += delta * delta2

            # Only compute std once we have enough observations
            if state["count"] >= 2:
                variance = state["M2"] / state["count"]
                state["std"] = math.sqrt(variance)
            else:
                state["std"] = 0.0

            baseline[channel] = state

            ## Log updated statistics for this channel ##
            logger.info(
                "Baseline updated — channel=%s count=%d mean=%.4f std=%.4f",
                channel, state["count"], state["mean"], state["std"]
            )

        except Exception as e:
            logger.error("Failed to update baseline for channel %s: %s", channel, e)
            raise

        return baseline

    def get_stats(self, baseline: dict, channel: str) -> Optional[dict]:
        ## Log baseline stat lookup for traceability ##
        stats = baseline.get(channel)
        if stats is None:
            logger.warning("No baseline stats found for channel: %s", channel)
        return stats
