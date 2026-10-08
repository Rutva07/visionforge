"""Command-line orchestration for the end-to-end pipeline."""
import argparse
import json
from pathlib import Path

from .config import Config
from .storage import make_storage
from .database import connect
from .pipeline import DATASETS, ingest, clean, deduplicate, materialize, inspect, import_s3_manifest
from .training import train_classifier, train_diffusion, sample_diffusion
from .report import update_readme, results_table


def parser():
    p = argparse.ArgumentParser(prog="visionforge", description="S3-backed image engineering pipeline")
    sub = p.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Verify metadata database and image object storage")
    init = sub.add_parser("init", help="Create database tables and test connectivity")
    scan = sub.add_parser("stats", help="Image and augmentation counts")
    scan.add_argument("--dataset", choices=DATASETS)
    ingest_cmd = sub.add_parser("ingest", help="Download official data and upload canonical images + metadata")
    ingest_cmd.add_argument("--dataset", choices=DATASETS, required=True)
    ingest_cmd.add_argument("--dataset-dir", default="datasets")
    ingest_cmd.add_argument("--limit", type=int, help="Maximum images per official split (debug)")
    manifest = sub.add_parser("import-manifest", help="Index already-uploaded S3 images using an authoritative CSV label manifest")
    manifest.add_argument("--csv", required=True, help="Columns: dataset,split,source_index,label,key")
    manifest.add_argument("--limit", type=int)
    clean_cmd = sub.add_parser("clean", help="Recheck S3 objects, quarantine invalid images")
    clean_cmd.add_argument("--dataset", choices=DATASETS, required=True)
    clean_cmd.add_argument("--limit", type=int)
    dedup = sub.add_parser("dedup", help="Exact SHA dedup and optional near-pHash train dedup")
    dedup.add_argument("--dataset", choices=DATASETS, required=True)
    dedup.add_argument("--near-threshold", type=int, default=0)
    aug = sub.add_parser("augment", help="Materialize versions of training images into S3")
    aug.add_argument("--dataset", choices=DATASETS, required=True)
    aug.add_argument("--variants", type=int, default=9)
    aug.add_argument("--limit", type=int, help="Max source training images (debug)")
    train = sub.add_parser("train", help="Train baseline or augmented PyTorch classifier")
    train.add_argument("--dataset", choices=DATASETS, required=True)
    train.add_argument("--mode", choices=("baseline", "augmented"), required=True)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--lr", type=float, default=.001)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--max-steps-per-epoch", type=int)
    train.add_argument("--limit", type=int, help="Debug: max samples for each partition")
    train.add_argument("--device", default="auto")
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--width", type=int, default=32)
    train.add_argument("--artifacts-dir", default="artifacts")
    ddpm = sub.add_parser("train-diffusion", help="Train conditional DDPM (augmented train set by default)")
    ddpm.add_argument("--dataset", choices=DATASETS, required=True)
    ddpm.add_argument("--mode", choices=("baseline", "augmented"), default="augmented")
    ddpm.add_argument("--epochs", type=int, default=10)
    ddpm.add_argument("--batch-size", type=int, default=64)
    ddpm.add_argument("--lr", type=float, default=.0002)
    ddpm.add_argument("--seed", type=int, default=42)
    ddpm.add_argument("--steps", type=int, default=200)
    ddpm.add_argument("--width", type=int, default=32)
    ddpm.add_argument("--limit", type=int)
    ddpm.add_argument("--max-steps-per-epoch", type=int)
    ddpm.add_argument("--device", default="auto")
    ddpm.add_argument("--artifacts-dir", default="artifacts")
    gen = sub.add_parser("sample-diffusion", help="Sample pretrained DDPM to an image contact sheet")
    gen.add_argument("--checkpoint", required=True)
    gen.add_argument("--count", type=int, default=16)
    gen.add_argument("--output")
    gen.add_argument("--device", default="auto")
    report = sub.add_parser("report", help="Update README with real comparable classification results")
    report.add_argument("--readme", default="README.md")
    return p


def run(args):
    if args.command == "sample-diffusion":
        return sample_diffusion(args.checkpoint, count=args.count, output=args.output, device=args.device)
    cfg = Config.from_env()
    maker = connect(cfg.database_url)
    with maker() as session:
        if args.command == "init":
            return {"database": "ready", "backend": cfg.storage_backend}
        if args.command == "report":
            return update_readme(session, args.readme)
        if args.command == "stats":
            return inspect(session, args.dataset)
        store = make_storage(cfg)
        if args.command == "doctor":
            return {"database": "ready", "storage": store.healthcheck()}
        if args.command == "ingest":
            return ingest(args.dataset, storage=store, session=session, config=cfg,
                          dataset_dir=args.dataset_dir, limit=args.limit)
        if args.command == "import-manifest":
            return import_s3_manifest(storage=store, session=session, manifest_path=args.csv, limit=args.limit)
        if args.command == "clean":
            return clean(storage=store, session=session, dataset=args.dataset, limit=args.limit)
        if args.command == "dedup":
            return deduplicate(session=session, dataset=args.dataset, near_threshold=args.near_threshold)
        if args.command == "augment":
            return materialize(storage=store, session=session, config=cfg,
                               dataset=args.dataset, variants=args.variants, limit=args.limit)
        if args.command == "train":
            return train_classifier(session=session, storage=store, config=cfg, name=args.dataset,
                                    mode=args.mode, epochs=args.epochs, batch_size=args.batch_size,
                                    lr=args.lr, seed=args.seed, max_steps_per_epoch=args.max_steps_per_epoch,
                                    limit=args.limit, device=args.device, workers=args.workers,
                                    width=args.width, artifacts_dir=args.artifacts_dir)
        if args.command == "train-diffusion":
            return train_diffusion(session=session, storage=store, config=cfg, name=args.dataset, mode=args.mode,
                                   epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                                   seed=args.seed, steps=args.steps, width=args.width,
                                   limit=args.limit, max_steps_per_epoch=args.max_steps_per_epoch,
                                   device=args.device, artifacts_dir=args.artifacts_dir)
    raise ValueError(args.command)


def main(argv=None):
    args = parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, default=str))


if __name__ == "__main__":
    main()
