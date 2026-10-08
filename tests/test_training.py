from unittest.mock import patch
from pathlib import Path
from sqlalchemy import select, func
import torch
from visionforge.pipeline import ingest, deduplicate, materialize
from visionforge.training import train_classifier, train_diffusion, sample_diffusion
from visionforge.models import Classifier, ConditionalUNet, Diffusion
from visionforge.database import Experiment
from visionforge.report import results_table, update_readme
from .test_pipeline import toy_dataset


def test_models_shapes():
    clf = Classifier(3, 100, width=8)
    assert clf(torch.randn(2, 3, 32, 32)).shape == (2, 100)
    for channels, size, classes in ((3, 32, 100), (1, 28, 10)):
        net = ConditionalUNet(channels, classes, width=8)
        noise = torch.randn(2, channels, size, size)
        labels = torch.tensor([0, 1])
        schedule = Diffusion(steps=3, device="cpu")
        assert net(noise, torch.tensor([1, 2]), labels).shape == noise.shape
        assert schedule.loss(net, noise, labels).ndim == 0
        assert schedule.sample(net, labels, channels, side=size).shape == noise.shape


def test_classifier_both_arms_diffusion_and_reporting(rig, tmp_path):
    cfg, store, session = rig
    with patch("visionforge.pipeline.load_official_dataset", side_effect=toy_dataset):
        ingest("fashionmnist", storage=store, session=session, config=cfg)
    deduplicate(session=session, dataset="fashionmnist")
    materialize(storage=store, session=session, config=cfg, dataset="fashionmnist", variants=1)
    artifacts = tmp_path / "artifacts"
    kwargs = dict(session=session, storage=store, config=cfg, name="fashionmnist",
                  epochs=1, batch_size=8, max_steps_per_epoch=1,
                  width=8, artifacts_dir=artifacts, device="cpu", limit=8)
    b = train_classifier(**kwargs, mode="baseline")
    a = train_classifier(**kwargs, mode="augmented")
    assert 0 <= b["test_accuracy"] <= 1
    assert 0 <= a["test_macro_f1"] <= 1
    assert a["train_images"] > b["train_images"]
    assert not a["full_test"]  # synthetic fixtures can never count as actual full dataset benchmarks
    assert session.scalar(select(func.count()).select_from(Experiment)) == 2
    assert "Not measured" in results_table(session)
    fake_readme = tmp_path / "README.md"
    fake_readme.write_text("Header\n<!-- RESULTS_START -->\ninitial\n<!-- RESULTS_END -->\nfooter")
    assert "Not measured" in update_readme(session, fake_readme)["table"]
    d = train_diffusion(session=session, storage=store, config=cfg, name="fashionmnist",
                        mode="augmented", epochs=1, batch_size=4, max_steps_per_epoch=1, width=8,
                        steps=3, limit=8, artifacts_dir=artifacts, device="cpu")
    image = sample_diffusion(d["checkpoint"], count=2, output=tmp_path / "samples.png", device="cpu")
    assert Path(image["output"]).is_file()


def test_report_only_matched_full_test_results(rig):
    cfg, store, session = rig
    from visionforge.database import Experiment
    common = dict(dataset="cifar100", seed=42, epochs=20,
                  best_val_accuracy=.6, train_images=45000, val_images=5000,
                  test_images=10000, full_test=True, checkpoint="model.pt",
                  configuration={"width": 32, "max_steps_per_epoch": 300,
                                 "lr": .001, "batch_size": 128})
    session.add(Experiment(**common, mode="baseline", test_accuracy=.40, test_macro_f1=.38))
    session.add(Experiment(**common, mode="augmented", test_accuracy=.45, test_macro_f1=.43))
    session.commit()
    result = results_table(session)
    assert "40.00%" in result and "45.00%" in result
    assert "0.3800" in result and "0.4300" in result
    assert "fashionmnist | Not measured" in result
