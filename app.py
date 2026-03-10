# app.py
import io
import json
import os
import boto3
import pandas as pd
import requests
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, Request
from baseline import BaselineManager
from processor import process_file

## Logging imports ##
import logging
import sys
## End logging imports ##

app = FastAPI(title="Anomaly Detection Pipeline")

s3 = boto3.client("s3")
BUCKET_NAME = os.environ["BUCKET_NAME"]

## Configure logging to file and stdout ##
LOG_FILE = "/opt/anomaly-detection/app.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)
logger.info("Application startup — BUCKET_NAME=%s", BUCKET_NAME)
## End logging setup ##


# ── SNS subscription confirmation + message handler ──────────────────────────

@app.post("/notify")
async def handle_sns(request: Request, background_tasks: BackgroundTasks):
    ## Wrap entire handler in try/except — SNS delivery failures are critical ##
    try:
        body = await request.json()
    except Exception as e:
        ## Log malformed request body ##
        logger.error("Failed to parse SNS request body: %s", e)
        return {"status": "error", "detail": "invalid JSON"}
    ## End request parse error handling ##

    msg_type = request.headers.get("x-amz-sns-message-type")

    if msg_type == "SubscriptionConfirmation":
        ## Log subscription confirmation attempt ##
        logger.info("SNS SubscriptionConfirmation received — confirming...")
        try:
            confirm_url = body["SubscribeURL"]
            requests.get(confirm_url, timeout=10)
            logger.info("SNS subscription confirmed successfully.")
        except Exception as e:
            logger.error("Failed to confirm SNS subscription: %s", e)
            return {"status": "error", "detail": "confirmation failed"}
        ## End confirmation error handling ##
        return {"status": "confirmed"}

    if msg_type == "Notification":
        ## Log incoming notification and parse S3 event ##
        try:
            s3_event = json.loads(body["Message"])
            for record in s3_event.get("Records", []):
                key = record["s3"]["object"]["key"]
                if key.startswith("raw/") and key.endswith(".csv"):
                    ## Log file arrival ##
                    logger.info("New file arrived — dispatching background task: %s", key)
                    background_tasks.add_task(process_file, BUCKET_NAME, key)
        except Exception as e:
            logger.error("Failed to parse SNS notification message: %s", e)
            return {"status": "error", "detail": "notification parse failed"}
        ## End notification error handling ##

    return {"status": "ok"}


# ── Query endpoints ───────────────────────────────────────────────────────────

@app.get("/anomalies/recent")
def get_recent_anomalies(limit: int = 50):
    """Return rows flagged as anomalies across the 10 most recent processed files."""
    ## Wrap S3 listing and reads in try/except ##
    try:
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix="processed/")

        keys = sorted(
            [
                obj["Key"]
                for page in pages
                for obj in page.get("Contents", [])
                if obj["Key"].endswith(".csv")
            ],
            reverse=True,
        )[:10]

        all_anomalies = []
        for key in keys:
            try:
                response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
                df = pd.read_csv(io.BytesIO(response["Body"].read()))
                if "anomaly" in df.columns:
                    flagged = df[df["anomaly"] == True].copy()
                    flagged["source_file"] = key
                    all_anomalies.append(flagged)
            except Exception as e:
                ## Log per-file read failures without aborting the whole request ##
                logger.error("Failed to read processed file %s: %s", key, e)
            ## End per-file error handling ##

    except Exception as e:
        logger.error("Failed to list processed files from S3: %s", e)
        return {"error": "Failed to retrieve anomalies", "detail": str(e)}
    ## End S3 listing error handling ##

    if not all_anomalies:
        return {"count": 0, "anomalies": []}

    combined = pd.concat(all_anomalies).head(limit)
    return {"count": len(combined), "anomalies": combined.to_dict(orient="records")}


@app.get("/anomalies/summary")
def get_anomaly_summary():
    """Aggregate anomaly rates across all processed files using their summary JSONs."""
    ## Wrap S3 listing and summary reads in try/except ##
    try:
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix="processed/")

        summaries = []
        for page in pages:
            for obj in page.get("Contents", []):
                if obj["Key"].endswith("_summary.json"):
                    try:
                        response = s3.get_object(Bucket=BUCKET_NAME, Key=obj["Key"])
                        summaries.append(json.loads(response["Body"].read()))
                    except Exception as e:
                        ## Log per-summary read failures without aborting ##
                        logger.error("Failed to read summary file %s: %s", obj["Key"], e)
                    ## End per-summary error handling ##

    except Exception as e:
        logger.error("Failed to list summary files from S3: %s", e)
        return {"error": "Failed to retrieve summary", "detail": str(e)}
    ## End S3 listing error handling ##

    if not summaries:
        return {"message": "No processed files yet."}

    total_rows = sum(s["total_rows"] for s in summaries)
    total_anomalies = sum(s["anomaly_count"] for s in summaries)

    return {
        "files_processed": len(summaries),
        "total_rows_scored": total_rows,
        "total_anomalies": total_anomalies,
        "overall_anomaly_rate": round(total_anomalies / total_rows, 4) if total_rows > 0 else 0,
        "most_recent": sorted(summaries, key=lambda x: x["processed_at"], reverse=True)[:5],
    }


@app.get("/baseline/current")
def get_current_baseline():
    """Show the current per-channel statistics the detector is working from."""
    ## Wrap baseline load in try/except ##
    try:
        baseline_mgr = BaselineManager(bucket=BUCKET_NAME)
        baseline = baseline_mgr.load()
    except Exception as e:
        logger.error("Failed to load baseline from S3: %s", e)
        return {"error": "Failed to load baseline", "detail": str(e)}
    ## End baseline load error handling ##

    channels = {}
    for channel, stats in baseline.items():
        if channel == "last_updated":
            continue
        channels[channel] = {
            "observations": stats["count"],
            "mean": round(stats["mean"], 4),
            "std": round(stats.get("std", 0.0), 4),
            "baseline_mature": stats["count"] >= 30,
        }

    return {
        "last_updated": baseline.get("last_updated"),
        "channels": channels,
    }


@app.get("/health")
def health():
    ## Log health check ##
    logger.info("Health check called.")
    ## End health check log ##
    return {"status": "ok", "bucket": BUCKET_NAME, "timestamp": datetime.utcnow().isoformat()}
