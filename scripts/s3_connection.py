#!/usr/bin/env python3
"""AWS S3 session examples: connect, list, get, put, presign (also works with MinIO).

Usage:
  python scripts/s3_connection.py health
  python scripts/s3_connection.py ls --prefix visionforge/raw/cifar100
  python scripts/s3_connection.py get --key visionforge/raw/cifar100/test/000000.png --out demo.png
  python scripts/s3_connection.py put --key visionforge/manual/example.png --file example.png
  python scripts/s3_connection.py url --key visionforge/manual/example.png

Use AWS_PROFILE or IAM task roles in AWS. Do not commit credentials.
"""
import argparse
import json
from pathlib import Path

from visionforge.config import Config
from visionforge.storage import make_storage


def main():
    p = argparse.ArgumentParser(description="Interact with existing S3 bucket")
    p.add_argument("operation", choices=("health", "ls", "get", "put", "url"))
    p.add_argument("--prefix", default="")
    p.add_argument("--key")
    p.add_argument("--file")
    p.add_argument("--out")
    p.add_argument("--expires", type=int, default=3600)
    args = p.parse_args()
    store = make_storage(Config.from_env())
    if args.operation == "health":
        result = store.healthcheck()
    elif args.operation == "ls":
        result = {"keys": list(store.iter_keys(args.prefix))}
    else:
        if not args.key:
            p.error("--key is required for get, put or url")
        if args.operation == "get":
            if not args.out:
                p.error("--out is required")
            Path(args.out).write_bytes(store.get(args.key))
            result = {"downloaded": args.key, "path": args.out}
        elif args.operation == "put":
            if not args.file:
                p.error("--file is required")
            store.put(args.key, Path(args.file).read_bytes())
            result = {"uploaded": args.key}
        else:
            result = {"presigned_url": store.presigned_url(args.key, args.expires)}
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
