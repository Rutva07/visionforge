"""Comparable classifier arms and class-conditional diffusion model training."""
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, RandomSampler

from .database import Experiment
from .loader import IndexedImages, image_channels
from .models import Classifier, ConditionalUNet, Diffusion
from .pipeline import DATASETS


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def datasets(session, storage, config, name, mode, limit):
    return [IndexedImages(session=session, storage=storage, cache_dir=config.cache_dir,
                          dataset=name, partition=p, mode=mode, limit=limit)
            for p in ("train", "val", "test")]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    truths, guesses = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        truths.extend(labels.tolist())
        guesses.extend(logits.argmax(dim=1).cpu().tolist())
    if not truths:
        raise ValueError("Evaluation partition is empty")
    return {"accuracy": float(accuracy_score(truths, guesses)),
            "macro_f1": float(f1_score(truths, guesses, average="macro", zero_division=0))}


def train_classifier(*, session, storage, config, name, mode, epochs=20, batch_size=128,
                     lr=0.001, seed=42, max_steps_per_epoch=None, limit=None,
                     device="auto", workers=0, width=32, artifacts_dir="artifacts"):
    if name not in DATASETS:
        raise ValueError(name)
    if mode not in ("baseline", "augmented"):
        raise ValueError(mode)
    set_seed(seed)
    train, val, test = datasets(session, storage, config, name, mode, limit)
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("One or more data splits are empty. Ingest more records first.")
    if mode == "augmented" and len(train) == len(IndexedImages(
        session=session, storage=storage, cache_dir=config.cache_dir,
        dataset=name, partition="train", mode="baseline", limit=limit)):
        raise ValueError("No materialized training variants found; run `visionforge augment` first")
    if max_steps_per_epoch is not None and max_steps_per_epoch < 1:
        raise ValueError("max_steps_per_epoch must be positive")
    sampler = (RandomSampler(train, replacement=True, num_samples=max_steps_per_epoch * batch_size)
               if max_steps_per_epoch is not None else None)
    loader_args = {"batch_size": batch_size, "num_workers": workers, "pin_memory": False,
                   "persistent_workers": workers > 0}
    train_loader = DataLoader(train, shuffle=sampler is None, sampler=sampler, **loader_args)
    val_loader = DataLoader(val, shuffle=False, **loader_args)
    test_loader = DataLoader(test, shuffle=False, **loader_args)
    target = device_for(device)
    model = Classifier(image_channels(name), DATASETS[name]["classes"], width=width).to(target)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val = -1.0
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / f"{name}_{mode}_seed{seed}.pt"
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss, total_n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(target), y.to(target)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(y)
            total_n += len(y)
        scheduler.step()
        validation = evaluate(model, val_loader, target)
        result = {"epoch": epoch + 1, "train_loss": total_loss / total_n,
                  "val_accuracy": validation["accuracy"], "val_macro_f1": validation["macro_f1"]}
        history.append(result)
        print(json.dumps(result), flush=True)
        if validation["accuracy"] >= best_val:
            best_val = validation["accuracy"]
            torch.save({"model": model.state_dict(), "dataset": name,
                        "classes": DATASETS[name]["classes"], "channels": image_channels(name),
                        "width": width, "mode": mode, "seed": seed}, checkpoint)
    # Test exactly once after selecting checkpoint using validation only.
    saved = torch.load(checkpoint, map_location=target, weights_only=True)
    model.load_state_dict(saved["model"])
    test_result = evaluate(model, test_loader, target)
    full_test = len(test) == DATASETS[name]["test"] and limit is None
    config_params = {"epochs": epochs, "batch_size": batch_size, "lr": lr,
                     "max_steps_per_epoch": max_steps_per_epoch, "width": width,
                     "limit_per_partition": limit, "workers": workers}
    exp = Experiment(dataset=name, mode=mode, seed=seed, epochs=epochs,
                     test_accuracy=test_result["accuracy"],
                     test_macro_f1=test_result["macro_f1"], best_val_accuracy=best_val,
                     train_images=len(train), val_images=len(val), test_images=len(test),
                     full_test=full_test, checkpoint=str(checkpoint), configuration=config_params)
    session.add(exp)
    session.commit()
    summary = {"experiment_id": exp.id, "dataset": name, "mode": mode,
               "test_accuracy": test_result["accuracy"], "test_macro_f1": test_result["macro_f1"],
               "best_val_accuracy": best_val, "train_images": len(train),
               "val_images": len(val), "test_images": len(test), "full_test": full_test,
               "checkpoint": str(checkpoint), "configuration": config_params,
               "history": history}
    (out_dir / f"{name}_{mode}_seed{seed}_metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


def train_diffusion(*, session, storage, config, name, mode="augmented", epochs=10, batch_size=64,
                    lr=0.0002, seed=42, steps=200, width=32, limit=None,
                    max_steps_per_epoch=None, device="auto", artifacts_dir="artifacts"):
    if name not in DATASETS:
        raise ValueError(name)
    set_seed(seed)
    train = IndexedImages(session=session, storage=storage, cache_dir=config.cache_dir,
                          dataset=name, partition="train", mode=mode, limit=limit)
    if mode == "augmented" and len(train) == len(IndexedImages(
        session=session, storage=storage, cache_dir=config.cache_dir,
        dataset=name, partition="train", mode="baseline", limit=limit)):
        raise ValueError("No materialized augmentations for DDPM; run `visionforge augment` first")
    if not train:
        raise ValueError("No training data")
    target = device_for(device)
    model = ConditionalUNet(image_channels(name), DATASETS[name]["classes"], width=width).to(target)
    diffusion = Diffusion(steps, target)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sampler = (RandomSampler(train, replacement=True, num_samples=max_steps_per_epoch * batch_size)
               if max_steps_per_epoch is not None else None)
    loader = DataLoader(train, batch_size=batch_size, sampler=sampler,
                        shuffle=sampler is None, num_workers=0)
    history = []
    for epoch in range(epochs):
        model.train()
        cumulative, batches = 0., 0
        for x, y in loader:
            x, y = x.to(target), y.to(target)
            opt.zero_grad(set_to_none=True)
            loss = diffusion.loss(model, x, y)
            loss.backward()
            opt.step()
            cumulative += loss.item()
            batches += 1
        item = {"epoch": epoch + 1, "noise_mse": cumulative / batches}
        history.append(item)
        print(json.dumps(item), flush=True)
    output = Path(artifacts_dir)
    output.mkdir(parents=True, exist_ok=True)
    dest = output / f"{name}_{mode}_ddpm_seed{seed}.pt"
    torch.save({"model": model.state_dict(), "dataset": name, "steps": steps, "width": width,
                "seed": seed, "mode": mode, "channels": image_channels(name), "classes": DATASETS[name]["classes"]}, dest)
    result = {"dataset": name, "mode": mode, "checkpoint": str(dest), "steps": steps,
              "history": history, "sample_images": None,
              "note": "DDPM training loss; not evidence of classifier improvement"}
    (output / f"{name}_{mode}_ddpm_seed{seed}_metrics.json").write_text(json.dumps(result, indent=2))
    return result


@torch.no_grad()
def sample_diffusion(checkpoint, *, count=16, output=None, device="auto"):
    if count < 1 or count > 128:
        raise ValueError("count must be in 1..128")
    target = device_for(device)
    saved = torch.load(checkpoint, map_location=target, weights_only=True)
    model = ConditionalUNet(saved["channels"], saved["classes"], width=saved["width"]).to(target)
    model.load_state_dict(saved["model"])
    diffuser = Diffusion(saved["steps"], target)
    labels = torch.arange(count, device=target) % saved["classes"]
    side = 28 if saved["dataset"] == "fashionmnist" else 32
    pictures = diffuser.sample(model, labels, saved["channels"], side=side).cpu()
    pictures = ((pictures + 1) * 127.5).round().clamp(0, 255).byte().numpy()
    columns = min(4, count)
    rows = (count + columns - 1) // columns
    grid = np.zeros((rows * side, columns * side, 3), dtype=np.uint8)
    for i, img in enumerate(pictures):
        if saved["channels"] == 3:
            image = cv2.cvtColor(img.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
        else:
            image = cv2.cvtColor(img[0], cv2.COLOR_GRAY2BGR)
        r, c = divmod(i, columns)
        grid[r * side:(r + 1) * side, c * side:(c + 1) * side] = image
    dest = Path(output) if output else Path(checkpoint).with_suffix(".png")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dest), grid):
        raise OSError(f"Could not write {dest}")
    return {"output": str(dest), "images": count, "labels": labels.cpu().tolist()}
