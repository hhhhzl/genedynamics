#!/usr/bin/env python3
"""Retrospective, grouped public-state observability check for T3 overforce."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
FEATURES = (
    "tcp_x", "tcp_y", "tcp_z", "lead_x", "lead_y", "normal_force",
    "normal_force_change", "tcp_x_change", "tcp_y_change", "tcp_z_change",
    "issued_vx", "issued_vy", "issued_vz", "issued_yaw",
    "issued_stiffness", "issued_force",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _feature(trace: dict, previous: dict) -> np.ndarray:
    """Use only public pose, measured normal force, reference and issued command."""
    pose = np.asarray(trace["tcp_position"], dtype=np.float64)
    old_pose = np.asarray(previous["tcp_position"], dtype=np.float64)
    lead = np.asarray(trace["public_reference_lead"], dtype=np.float64)
    action = np.asarray(trace["issued_action"], dtype=np.float64)
    return np.asarray((
        pose[0], pose[1], pose[2], lead[0], lead[1],
        trace["normal_force"], trace["normal_force"] - previous["normal_force"],
        *(pose - old_pose), action[0], action[1], action[2], action[5],
        action[6], action[12],
    ), dtype=np.float64)


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    return float((positives[:, None] > negatives).mean()
                 + .5 * (positives[:, None] == negatives).mean())


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    ordered = labels[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(np.sum(precision * ordered) / ordered.sum())


def _evaluate(labels: np.ndarray, scores: np.ndarray, threshold: float,
              episode_ids: np.ndarray, outcomes: dict[str, bool]) -> dict:
    predicted = scores >= threshold
    ids = list(dict.fromkeys(episode_ids))
    alerts = {key: bool(np.any(predicted[episode_ids == key])) for key in ids}
    failed = [key for key in ids if outcomes[key]]
    succeeded = [key for key in ids if not outcomes[key]]
    return {
        "windows": len(labels), "next_tick_overforce_events": int(labels.sum()),
        "prevalence": float(labels.mean()), "auroc": _auc(labels, scores),
        "average_precision": _average_precision(labels, scores),
        "window_fpr": float(predicted[labels == 0].mean()),
        "window_recall": float(predicted[labels == 1].mean()),
        "episodes_with_overforce": len(failed),
        "overforce_episodes_alerted": sum(alerts[key] for key in failed),
        "successful_episodes_alerted": sum(alerts[key] for key in succeeded),
        "successful_episodes": len(succeeded),
    }


def analyze(report_path: Path, output: Path) -> dict:
    report = json.loads(report_path.read_text())
    if (report.get("status") != "complete_development_diagnosis"
            or report.get("physical_episodes") != 96 or len(report["rows"]) != 96):
        raise ValueError("complete frozen T3 paired diagnosis required")
    data = []
    outcomes = {}
    for row in report["rows"]:
        evidence_path = ROOT / row["evidence"]["path"]
        if digest(evidence_path) != row["evidence"]["sha256"]:
            raise ValueError("T3 evidence SHA differs")
        evidence = json.loads(gzip.decompress(evidence_path.read_bytes()))
        traces = evidence["records"][0]["traces"]
        episode_id = f"{row['cell']}:{row['slot']}:{row['mode']}:{row['method']}"
        outcomes[episode_id] = row["outcome"] == "overforce"
        for tick in range(20, min(len(traces) - 1, 121)):
            current = traces[tick]
            if not .594 <= current["tcp_position"][2] <= .5995:
                continue
            feature = _feature(current, traces[tick - 1])
            if not np.isfinite(feature).all():
                raise ValueError("nonfinite public feature")
            data.append((row["slot"] < 4, episode_id, feature,
                         int(outcomes[episode_id] and tick == len(traces) - 2)))
    train = [sample for sample in data if sample[0]]
    test = [sample for sample in data if not sample[0]]
    x_train = np.stack([sample[2] for sample in train])
    y_train = np.asarray([sample[3] for sample in train], dtype=np.int64)
    x_test = np.stack([sample[2] for sample in test])
    y_test = np.asarray([sample[3] for sample in test], dtype=np.int64)
    if not (20 <= y_train.sum() <= 40 and 15 <= y_test.sum() <= 35):
        raise ValueError("T3 train/test event prevalence changed")
    mean = x_train.mean(0)
    std = x_train.std(0).clip(min=1e-6)
    torch.manual_seed(0)
    weight = torch.nn.Parameter(torch.zeros(len(FEATURES), dtype=torch.float64))
    bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.Adam((weight, bias), lr=.03)
    features = torch.as_tensor((x_train - mean) / std, dtype=torch.float64)
    labels = torch.as_tensor(y_train, dtype=torch.float64)
    positive_weight = float((len(y_train) - y_train.sum()) / y_train.sum())
    for _ in range(800):
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            features @ weight + bias, labels,
            pos_weight=torch.tensor(positive_weight, dtype=torch.float64),
        ) + .02 * weight.square().sum()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        learned_train = (features @ weight + bias).numpy()
        learned_test = (
            torch.as_tensor((x_test - mean) / std, dtype=torch.float64) @ weight + bias
        ).numpy()
    lead_train = np.linalg.norm(x_train[:, 3:5], axis=1)
    lead_test = np.linalg.norm(x_test[:, 3:5], axis=1)
    comparisons = {}
    for name, scores_train, scores_test in (
        ("public_linear_risk", learned_train, learned_test),
        ("reference_lead_norm", lead_train, lead_test),
    ):
        threshold = float(np.quantile(scores_train[y_train == 0], .95))
        comparisons[name] = {
            "train_negative_95pct_threshold": threshold,
            "train": _evaluate(y_train, scores_train, threshold,
                               np.asarray([sample[1] for sample in train]), outcomes),
            "heldout": _evaluate(y_test, scores_test, threshold,
                                 np.asarray([sample[1] for sample in test]), outcomes),
        }
    result = {
        "schema_version": "t3-public-impulse-observability-v1",
        "status": "retrospective_development_only", "physical_calls": 0, "gpu_calls": 0,
        "input_report_sha256": digest(report_path),
        "source_sha256": digest(Path(__file__)),
        "split": "slots 0-3 fit, slots 4-7 held out; paired methods stay together",
        "risk_window": "previous public control state, tick 20-120, TCP z 0.594-0.5995 m",
        "label": "next physical control tick is the first overforce termination",
        "features": FEATURES,
        "model": "weighted regularized linear logit; training negatives set 5% window false-alert threshold",
        "scope": "observability only; alerts do not establish safe interventions or a substep force certificate",
        "comparisons": comparisons,
    }
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.report, args.output)
    print(json.dumps({"status": result["status"], "comparisons": result["comparisons"]}, indent=2))


if __name__ == "__main__":
    main()
