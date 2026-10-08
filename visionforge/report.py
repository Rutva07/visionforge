"""Reproducible experimental comparison: never synthesize missing performance data."""
from pathlib import Path
import json
from sqlalchemy import select

from .database import Experiment

START = "<!-- RESULTS_START -->"
END = "<!-- RESULTS_END -->"


def results_table(session):
    experiments = list(session.scalars(select(Experiment).where(
        Experiment.full_test.is_(True)).order_by(Experiment.id.desc())))
    lines = ["| Dataset | Before: clean-only accuracy | After: augmented accuracy | Before macro-F1 | After macro-F1 |",
             "|---|---:|---:|---:|---:|"]
    for ds in ("cifar100", "fashionmnist"):
        modes = {}
        # Paired experiments must use the same seed, width, epochs and step budget.
        rows = [e for e in experiments if e.dataset == ds]
        pairs = []
        for base in rows:
            if base.mode != "baseline":
                continue
            for augmented in rows:
                if augmented.mode != "augmented":
                    continue
                comparable = (base.seed == augmented.seed and base.epochs == augmented.epochs and
                              all(base.configuration.get(k) == augmented.configuration.get(k)
                                  for k in ("width", "max_steps_per_epoch", "lr", "batch_size")))
                if comparable:
                    pairs.append((base, augmented))
        if pairs:
            base, aug = pairs[0]  # latest baseline with matching augmented run
            lines.append(f"| {ds} | {base.test_accuracy * 100:.2f}% | {aug.test_accuracy * 100:.2f}% | "
                         f"{base.test_macro_f1:.4f} | {aug.test_macro_f1:.4f} |")
        else:
            lines.append(f"| {ds} | Not measured | Not measured | Not measured | Not measured |")
    lines.append("\n*Only matched, full-official-test-set runs appear here. Debug runs and incomplete pairs are excluded. Accuracy improvements are not assumed.*")
    return "\n".join(lines)


def update_readme(session, readme="README.md"):
    path = Path(readme)
    text = path.read_text()
    if START not in text or END not in text:
        raise ValueError("README result markers missing")
    new = text.split(START, 1)[0] + START + "\n" + results_table(session) + "\n" + END + text.split(END, 1)[1]
    path.write_text(new)
    return {"updated": str(path), "table": results_table(session)}
