"""PostgreSQL-first metadata schema; SQLite works for self-contained smoke tests."""
from datetime import datetime, timezone
from sqlalchemy import (
    create_engine, DateTime, ForeignKey, Float, Index, Integer, String,
    JSON, UniqueConstraint, select, func
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


class Base(DeclarativeBase):
    pass


class ImageAsset(Base):
    __tablename__ = "image_assets"
    __table_args__ = (
        UniqueConstraint("dataset", "split", "source_index", name="uq_dataset_split_index"),
        Index("idx_asset_filter", "dataset", "split", "status"),
        Index("idx_asset_sha", "dataset", "sha256"),
        Index("idx_asset_phash", "dataset", "phash"),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset: Mapped[str] = mapped_column(String(48), nullable=False)
    split: Mapped[str] = mapped_column(String(12), nullable=False)
    source_index: Mapped[int] = mapped_column(Integer, nullable=False)
    label: Mapped[int] = mapped_column(Integer, nullable=False)
    key: Mapped[str] = mapped_column(String(512), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    phash: Mapped[str] = mapped_column(String(16), nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    channels: Mapped[int] = mapped_column(Integer, nullable=False)
    blur_score: Mapped[float] = mapped_column(Float, nullable=False)
    pixel_std: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="ready", nullable=False)
    reason: Mapped[str | None] = mapped_column(String(128))
    duplicate_of_id: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class Augmentation(Base):
    __tablename__ = "augmentations"
    __table_args__ = (
        UniqueConstraint("asset_id", "recipe_version", "variant_no", name="uq_variant"),
        Index("idx_variant_asset", "asset_id", "status"),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey("image_assets.id", ondelete="CASCADE"), nullable=False)
    recipe_version: Mapped[str] = mapped_column(String(32), nullable=False)
    variant_no: Mapped[int] = mapped_column(Integer, nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    key: Mapped[str] = mapped_column(String(512), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="ready", nullable=False)


class Experiment(Base):
    __tablename__ = "experiments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset: Mapped[str] = mapped_column(String(48), nullable=False)
    mode: Mapped[str] = mapped_column(String(24), nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    epochs: Mapped[int] = mapped_column(Integer, nullable=False)
    test_accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    test_macro_f1: Mapped[float] = mapped_column(Float, nullable=False)
    best_val_accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    train_images: Mapped[int] = mapped_column(Integer, nullable=False)
    val_images: Mapped[int] = mapped_column(Integer, nullable=False)
    test_images: Mapped[int] = mapped_column(Integer, nullable=False)
    full_test: Mapped[bool] = mapped_column(default=False, nullable=False)
    checkpoint: Mapped[str] = mapped_column(String(512), nullable=False)
    configuration: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


def connect(url: str):
    engine = create_engine(url, pool_pre_ping=True, future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)


def asset_counts(session):
    rows = session.execute(
        select(ImageAsset.dataset, ImageAsset.split, ImageAsset.status, func.count())
        .group_by(ImageAsset.dataset, ImageAsset.split, ImageAsset.status)
        .order_by(ImageAsset.dataset, ImageAsset.split, ImageAsset.status)
    ).all()
    return [{"dataset": d, "split": s, "status": st, "count": count} for d, s, st, count in rows]
