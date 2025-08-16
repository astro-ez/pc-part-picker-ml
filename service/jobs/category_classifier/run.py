import os, sys, time, argparse, joblib
from datetime import datetime, timezone
from typing import List, Dict, Any
import pandas as pd
import logging
from pymongo import MongoClient, UpdateOne
from pymongo.errors import PyMongoError


def iso_now():
    return datetime.now().isoformat()


def parse_args():
    parser = argparse.ArgumentParser(description="Category Classifier")
    parser.add_argument(
        "--mongo_uri",
        type=str,
        default=os.getenv("MONGO_URI", "mongodb://localhost:27017/"),
        help="MongoDB URI",
    )
    
    parser.add_argument(
        "--db",
        type=str,
        default=os.getenv("DB_NAME"),
        help="Database name",
    )

    parser.add_argument(
        "--collection",
        type=str,
        default=os.getenv("DB_COLLECTION"),
        help="Database collection name",
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.getenv("MODEL_PATH"),
        help="Path to the model file",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("BATCH_SIZE", 1000)),
        help="Batch size for processing",
    )

    parser.add_argument(
        "--features",
        nargs="+",
        default=os.getenv("FEATURES", "").split(","),
        help="List of features to use for prediction",
    )
    
    parser.add_argument(
        "--stale-after-hours",
        type=int,
        default=int(os.getenv("STALE_AFTER_HOURS", 0)),
        help="Number of hours after which data is considered stale",
    )

    parser.add_argument(
        "--logging",
        type=str,
        default=os.getenv("LOGGING_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    
    return parser.parse_args()

def build_query(stale_hours: int) -> Dict[str, Any]:
    query = {"need_scoring": True}
    if stale_hours > 0:
        stale_time = datetime.now(timezone.utc).timestamp() - (stale_hours * 3600)
        query = {"$or": [{"last_updated": {"$lt": stale_time}}, {"need_scoring": True}]}
    return query

def project_fields(feature_cols: List[str]) -> Dict[str, int]:
    proj = {col: 1 for col in feature_cols}
    proj.update({"_id": 1})
    return proj

def to_dataframe(docs: List[Dict[str, Any]], feature_cols: List[str]) -> pd.DataFrame:
    # Normalize; missing -> empty string
    rows = []
    for d in docs:
        row = {col: d.get(col, "") for col in feature_cols}
        row["_id"] = d["_id"]
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    args = parse_args()
    print(f"[{iso_now()}] Loading model from {args.model_path}", flush=True)
    pipe = joblib.load(args.model_path)

    logging.basicConfig(level=args.logging.upper(), format="%(asctime)s - %(levelname)s - %(message)s")

    # optional: set a version attribute in training code and persist
    model_version = getattr(pipe, "version_", os.path.basename(args.model_path))

    client = MongoClient(args.mongo_uri, retryWrites=True)
    coll = client[args.db][args.collection]

    query = build_query(args.stale_after_hours)
    projection = project_fields(args.features)

    processed = 0
    last_id = None

    while True:
        q = dict(query)
        if last_id is not None:
            # paginate by _id to avoid skipping when docs change
            q["_id"] = {"$gt": last_id}

        cursor = coll.find(q, projection=projection, sort=[("_id", 1)], batch_size=args.batch_size).limit(args.batch_size)
        docs = list(cursor)
        if not docs:
            break

        df = to_dataframe(docs, args.features)
        X = df[args.features]

        # Transform Input
        if hasattr(pipe, "transform"):
            X = pipe.transform(X)

        # Predict
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X)
            # if multiclass, take argmax
            y_idx = proba.argmax(axis=1)
            preds = [pipe.classes_[i] for i in y_idx]
            confidences = [float(proba[i, y_idx[i]]) for i in range(len(y_idx))]
        else:
            preds = pipe.predict(X)
            confidences = [None] * len(preds)

        scored_at = iso_now()

        ops = []
        for doc, pred, conf in zip(docs, preds, confidences):
            update = {
                "$set": {
                    "category": str(pred),
                    "category_proba": float(conf) if conf is not None else None,
                    "model_version": model_version,
                    "scored_at": scored_at,
                },
                "$unset": {"needs_scoring": ""}  # mark as done
            }
            ops.append(UpdateOne({"_id": doc["_id"]}, update, upsert=False))

        try:
            if ops:
                res = coll.bulk_write(ops, ordered=False)
                processed += res.matched_count
                print(f"[{iso_now()}] Updated {res.modified_count} docs (batch size {len(ops)})", flush=True)
        except PyMongoError as e:
            # minimal retry on batch failure: fall back to single updates
            print(f"[{iso_now()}] bulk_write failed: {e}. Retrying singly...", file=sys.stderr, flush=True)
            for op in ops:
                try:
                    coll.update_one(op._filter, op._doc, upsert=False)
                except PyMongoError as e2:
                    print(f"Failed updating {op._filter['_id']}: {e2}", file=sys.stderr)

        last_id = docs[-1]["_id"]

        if args.max_docs and processed >= args.max_docs:
            print(f"[{iso_now()}] Reached max_docs={args.max_docs}, stopping.", flush=True)
            break

    print(f"[{iso_now()}] DONE. Processed ~{processed} documents.", flush=True)

if __name__ == "__main__":
    main()