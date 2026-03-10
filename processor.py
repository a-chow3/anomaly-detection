#!/usr/bin/env python3
import json
import io
import boto3
import pandas as pd
from datetime import datetime
from baseline import BaselineManager
from detector import AnomalyDetector

## Get shared logger ##
import logging
logger = logging.getLogger(__name__)

s3 = boto3.client("s3")
NUMERIC_COLS = ["temperature", "humidity", "pressure", "wind_speed"]

def process_file(bucket: str, key: str):
    ## Wrap entire pipeline in try/except — this runs as a background task ##
    try:
        logger.info("Processing started: s3://%s/%s", bucket, key)

        # 1. Download raw file
        ## Wrap S3 download in try/except ##
        try:
            response = s3.get_object(Bucket=bucket, Key=key)
            df = pd.read_csv(io.BytesIO(response["Body"].read()))
            logger.info("Loaded %d rows, columns: %s", len(df), list(df.columns))
        except Exception as e:
            logger.error("Failed to download or parse file s3://%s/%s: %s", bucket, key, e)
            raise

        # 2. Load current baseline
        baseline_mgr = BaselineManager(bucket=bucket)
        baseline = baseline_mgr.load()

        # 3. Update baseline with values from this batch BEFORE scoring
        ## Wrap per-column baseline update in try/except ##
        for col in NUMERIC_COLS:
            if col in df.columns:
                try:
                    clean_values = df[col].dropna().tolist()
                    if clean_values:
                        baseline = baseline_mgr.update(baseline, col, clean_values)
                except Exception as e:
                    logger.error("Failed to update baseline for column %s: %s", col, e)

        # 4. Run detection
        ## Wrap detection in try/except ##
        try:
            detector = AnomalyDetector(z_threshold=3.0, contamination=0.05)
            scored_df = detector.run(df, NUMERIC_COLS, baseline, method="both")
            logger.info("Detection complete for %s", key)
        except Exception as e:
            logger.error("Anomaly detection failed for %s: %s", key, e)
            raise

        # 5. Write scored file to processed/ prefix
        ## Wrap scored CSV write in try/except ##
        try:
            output_key = key.replace("raw/", "processed/")
            csv_buffer = io.StringIO()
            scored_df.to_csv(csv_buffer, index=False)
            s3.put_object(
                Bucket=bucket,
                Key=output_key,
                Body=csv_buffer.getvalue(),
                ContentType="text/csv"
            )
            logger.info("Scored file written to s3://%s/%s", bucket, output_key)
        except Exception as e:
            logger.error("Failed to write scored file to S3 for %s: %s", key, e)
            raise

        # 6. Save updated baseline back to S3
        baseline_mgr.save(baseline)

        # 7. Build and return a processing summary
        anomaly_count = int(scored_df["anomaly"].sum()) if "anomaly" in scored_df else 0
        summary = {
            "source_key": key,
            "output_key": output_key,
            "processed_at": datetime.utcnow().isoformat(),
            "total_rows": len(df),
            "anomaly_count": anomaly_count,
            "anomaly_rate": round(anomaly_count / len(df), 4) if len(df) > 0 else 0,
            "baseline_observation_counts": {
                col: baseline.get(col, {}).get("count", 0) for col in NUMERIC_COLS
            }
        }

        # Write summary JSON alongside the processed file
        ## Wrap summary JSON write in try/except ##
        try:
            summary_key = output_key.replace(".csv", "_summary.json")
            s3.put_object(
                Bucket=bucket,
                Key=summary_key,
                Body=json.dumps(summary, indent=2),
                ContentType="application/json"
            )
            logger.info("Summary written to s3://%s/%s", bucket, summary_key)
        except Exception as e:
            logger.error("Failed to write summary JSON for %s: %s", key, e)
            raise

        logger.info("Processing complete: %d/%d anomalies flagged in %s", anomaly_count, len(df), key)
        return summary

    except Exception as e:
        logger.error("process_file failed for s3://%s/%s: %s", bucket, key, e)
        raise
