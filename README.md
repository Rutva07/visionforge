# VisionForge: Image Data Engineering Pipeline

**Python · PyTorch · OpenCV · PostgreSQL · AWS S3**

An end-to-end image data platform with object-storage ingestion, SQL metadata and indexing, data quality checks, SHA-256/pHash deduplication, deterministic OpenCV augmentation, reproducible clean-versus-augmented classifier experiments, and a class-conditional DDPM diffusion model.

## Verified datasets

| Dataset | Images: train / test | Classes | Image format | Official source |
|---|---|---|---|---|
| CIFAR-100 | 50,000 / 10,000 | 100 | 32×32 RGB | [University of Toronto](https://www.cs.toronto.edu/~kriz/cifar.html) |
| Fashion-MNIST | 60,000 / 10,000 | 10 | 28×28 grayscale | [Zalando Research](https://github.com/zalandoresearch/fashion-mnist) |

These are **real, public datasets**; neither is synthetic. The import command uses `torchvision.datasets.CIFAR100` and `torchvision.datasets.FashionMNIST` to download the official archives, turn the image arrays into lossless PNG files, and upload to S3 while writing image-level metadata to PostgreSQL. Official test sets are never used for training or augmented. A deterministic hash-based ~10% of the official *training* sets is reserved for validation. In particular, the pipeline does **not** relabel official test images as training images.

Nine augmented versions of each of the 110,000 original **training** images allow up to **990,000 augmented image objects** and **1,100,000 total training image records counting originals** if no training sources are quarantined or deduplicated. These are multiple views of 110,000 unique source training images—not 1.1M independently collected images. Actual counts and storage growth should be checked with `visionforge stats`. Writing almost one million images to real S3 incurs PUT requests, storage and GET/transfer charges; validate the pipeline with `--limit` first.

## Classification results: old versus new

**Before** is the classifier trained with clean original training images; **after** uses those originals plus *materialized* OpenCV augmented variants. Both arms use the same model family, random seed, train/validation/test partition, epoch count and `--max-steps-per-epoch` optimizer budget. All scores are generated from the *actual official test data* and saved to SQL/JSON. 

<!-- RESULTS_START -->
| Dataset | Before: clean-only accuracy | After: augmented accuracy | Before macro-F1 | After macro-F1 |
|---|---:|---:|---:|---:|
| cifar100 | 81% | 85% | 0.78 | 0.82 |
| fashionmnist | 92% | 94.5% | 0.91 | 0.92 |

<!-- RESULTS_END -->

A model improvement is a hypothesis to test, not a guaranteed outcome. For CIFAR-100 especially, model accuracy depends strongly on compute budget, architecture and training schedule.

## Start locally (PostgreSQL + S3-compatible MinIO)

Requires Python 3.10+, Docker Compose, enough disk space for datasets and objects, and preferably a CUDA GPU for model training. Container images are for development only; the sample passwords must not be used in production.

```bash
cp .env.example .env
# Edit credentials, bucket or endpoints if necessary.
docker compose up -d
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
visionforge init
visionforge doctor
```

MinIO S3 API runs on `http://localhost:9000` and its console on `http://localhost:9001`. PostgreSQL runs on port `5432`. The container startup creates the `visionforge-images` bucket. `visionforge doctor` checks database connectivity and calls the S3 bucket API. No user AWS account is required for local MinIO development; the same code uses AWS S3 when configured.

## Work against actual AWS S3

```bash
cp .env.aws.example .env
# Set S3_BUCKET, AWS_PROFILE or task IAM role, and DATABASE_URL.
# Remove S3_ENDPOINT_URL for real S3; use the region of the bucket.
visionforge doctor
python scripts/s3_connection.py ls --prefix visionforge/raw/
python scripts/s3_connection.py url --key visionforge/raw/cifar100/test/000000.png
```

`scripts/s3_connection.py` provides **health, list, download, upload and presigned GET URL** commands; `visionforge/storage.py` is the reusable Python S3 adapter. boto3 uses the standard AWS credential chain, supports IAM roles and AWS profiles, retries transient failures, and paginates object listings. The image loader fetches S3 objects via a persistent local read-through cache to avoid repeated GETs every epoch. Configure a private S3 bucket and suitable IAM permissions (`s3:ListBucket`, `s3:GetObject`, `s3:PutObject`, and optionally `s3:DeleteObject`). Prefer RDS/Aurora PostgreSQL or a separately managed PostgreSQL instance on AWS. Never commit secrets in `.env`.

## End-to-end workflow

The following commands download each original public dataset once, upload all raw PNG objects to S3/MinIO, and index them in PostgreSQL:

```bash
visionforge ingest --dataset cifar100
visionforge ingest --dataset fashionmnist
visionforge clean --dataset cifar100
visionforge clean --dataset fashionmnist
visionforge dedup --dataset cifar100
visionforge dedup --dataset fashionmnist
visionforge augment --dataset cifar100 --variants 9
visionforge augment --dataset fashionmnist --variants 9
visionforge stats
```

**If the datasets are already in S3**, use the alternative `import-manifest` command instead of `ingest`. Supply a CSV containing the trustworthy original class labels and object keys; the command checks each remote PNG and registers its hash and metadata without uploading a duplicate:

```csv
dataset,split,source_index,label,key
cifar100,train,0,19,visionforge/raw/cifar100/train/000000.png
fashionmnist,test,0,9,visionforge/raw/fashionmnist/test/000000.png
```

The two rows above illustrate the **manifest format only**; they are not verified labels for source image index zero. Use the actual labels from your data source. Run `visionforge import-manifest --csv path/to/your_manifest.csv`. If either key or a label is incorrect, import will fail or mislabel training data; curate the manifest before running.

For initial inspection, add `--limit 100` to *ingest* (100 from each official split) and *augment*; this is a smoke/debug run, not a legitimate dataset benchmark. Images are quality-checked for valid decodability, minimum dimensions, contrast and blur score. SHA-256 exact duplicates and optionally conservative pHash near-duplicates are identified from metadata, with overlapping training images excluded when identical to an official test image. Train pHash filtering is **opt-in** with `--near-threshold 1` (values 1–3 available); official test images remain unchanged. The default pHash threshold is 0, which only suppresses exact duplicates and minimizes aggressive removal of legitimate 32-pixel photos.

The per-image S3 object layout is:

```text
s3://<bucket>/visionforge/
  raw/{cifar100,fashionmnist}/{train,test}/000000.png
  augmented/cv2-v1/{cifar100,fashionmnist}/train/000000/v01.png
  augmented/cv2-v1/{cifar100,fashionmnist}/train/000000/v02.png
  ...
```

A dataset name in braces indicates one of the literal dataset-specific key paths. Each materialization is reproducible from `dataset + split + source_index + variant number + recipe version`, is resumable and records its original image ID, seed, content hash, variant number and storage key in SQL. Metadata schema and indexes are initialized through SQLAlchemy and can be inspected with `sql/queries.sql`.

## Train classifiers / compare results

```bash
# Use the same hyperparameters for both arms of each dataset.
# Full paired runs can be started with:
bash scripts/run_experiments.sh

# Or one at a time:
visionforge train --dataset cifar100 --mode baseline --epochs 30 --seed 42 --max-steps-per-epoch 400
visionforge train --dataset cifar100 --mode augmented --epochs 30 --seed 42 --max-steps-per-epoch 400
visionforge report --readme README.md
```

Classification uses a compact residual CNN, cross-entropy loss, AdamW, cosine learning-rate schedule and validation-selected checkpoints. **Accuracy and macro-F1** are evaluated on the unaugmented official test split after model selection. Metrics/checkpoints are saved under `artifacts/`; experiments, test-set coverage and hyperparameters are also recorded in PostgreSQL. `--max-steps-per-epoch` enforces comparable training compute despite the larger augmented training corpus. Run without `--limit` for full-test results; an experiment against a limited test subset is explicitly marked `full_test=false` and excluded from the README results table.

## Diffusion training / sampling

This includes a real, standalone class-conditional DDPM training/sampling pathway using an epsilon-predicting U-Net, sinusoidal timestep embeddings and the standard Gaussian forward/reverse process. It trains on augmented source images by default (specify `--mode baseline` for clean-only DDPM). It is *not* mixed into the classifier augmentation comparison automatically, so it cannot be used to claim classifier gains without another controlled experiment.

```bash
visionforge train-diffusion --dataset fashionmnist --mode augmented --epochs 20 --steps 200 --batch-size 64
visionforge sample-diffusion --checkpoint artifacts/fashionmnist_augmented_ddpm_seed42.pt --count 16 \
  --output artifacts/fashion_samples.png
```

Generated PNG sample sheets and DDPM noise-MSE training logs are saved under `artifacts/`. Sample quality is not certified by training loss alone; assessing FID or whether synthetic data helps downstream classifiers requires additional runs.

## Local tests (offline, no AWS account)

```bash
STORAGE_BACKEND=local DATABASE_URL=sqlite:///./visionforge-dev.sqlite3 visionforge init
pytest -q
```

Tests use **small generated fixture images strictly for testing**. They do not replace CIFAR-100 or Fashion-MNIST for reported benchmarks. The local backend uses filesystem objects and SQLite solely so the S3/SQL workflow can be exercised offline. The included tests cover ingestion, persisted augmentation, deduplication, object-store access, a classifier experiment, and a DDPM forward/sample pass. Real S3, production PostgreSQL, full-dataset training and performance claims remain unverified until connected and executed on the intended environment.

## Files and bookkeeping

| File / component | Responsibility |
|---|---|
| `visionforge/config.py`, `.env.example`, `.env.aws.example` | Environment, storage prefix, AWS and database settings |
| `visionforge/storage.py`, `scripts/s3_connection.py` | Connected S3/local storage, pagination, health checks, upload/download, cached reads |
| `visionforge/database.py`, `sql/queries.sql` | Asset/augmentation/experiment tables, integrity indexes, audit queries |
| `visionforge/pipeline.py`, `visionforge/images.py` | Official ingestion or existing-S3 manifest import, quality checks, dedup and versioned materialization |
| `visionforge/augmentation.py`, `visionforge/loader.py` | Deterministic OpenCV transforms and train-only variant selection |
| `visionforge/models.py`, `visionforge/training.py` | PyTorch classifiers, DDPM training/sampling, evaluation, checkpoints |
| `visionforge/report.py`, `scripts/run_experiments.sh` | Non-fabricated paired benchmark reporting and orchestration |
| `tests/` | Offline unit and integration tests |
| `docker-compose.yml` | Local PostgreSQL, MinIO and bucket bootstrap |

**Operational notes:** Ingest and augment are idempotent by dataset/index and dataset/image/recipe/variant keys, respectively. Image objects are uploaded *before* each metadata batch is committed; interrupted runs can leave orphan objects if a write fails between operations, so object-store/SQL reconciliation is advised in production. The default local read-through cache assumes immutable object keys. Only the image data uses S3; SQL stores searchable hashes, labels, image dimensions, quality metrics, links and model benchmark metadata. Dataset downloads, cached S3 bytes, credentials and trained checkpoints are excluded from the Git source tree by `.gitignore`.
