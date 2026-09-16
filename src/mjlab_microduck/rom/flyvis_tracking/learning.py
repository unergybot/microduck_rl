"""Supervised readout learning with training-only normalization and safe NPZ weights."""

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class Readout:
    def __init__(self, mean, scale, model):
        self.mean, self.scale, self.model = mean, scale, model.eval()
        self.metadata = {}

    def predict_batch(self, x):
        x = np.asarray(x, np.float32)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError("invalid readout features")
        with torch.inference_mode():
            return (
                self.model(torch.from_numpy((x - self.mean) / self.scale))
                .argmax(-1)
                .cpu()
                .numpy()
            )

    def save(self, path, metadata):
        values = {
            name: value.detach().cpu().numpy()
            for name, value in self.model.state_dict().items()
        }
        np.savez_compressed(
            path,
            mean=self.mean,
            scale=self.scale,
            metadata=json.dumps(metadata, allow_nan=False),
            **values,
        )
        self.metadata = metadata

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as data:
            model = nn.Sequential(
                nn.Linear(len(data["mean"]), 64), nn.ReLU(), nn.Linear(64, 4)
            )
            model.load_state_dict(
                {
                    name: torch.from_numpy(data[name].copy())
                    for name in model.state_dict()
                }
            )
            result = cls(data["mean"].copy(), data["scale"].copy(), model)
            result.metadata = json.loads(str(data["metadata"]))
            return result


def fit_readout(x, y, validation_x, validation_y, *, seed, epochs=50):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    x, validation_x = np.asarray(x, np.float32), np.asarray(validation_x, np.float32)
    y, validation_y = np.asarray(y, np.int64), np.asarray(validation_y, np.int64)
    if (
        len(x) == 0
        or len(validation_x) == 0
        or not np.isfinite(x).all()
        or not np.isfinite(validation_x).all()
        or not np.isin(y, range(4)).all()
        or not np.isin(validation_y, range(4)).all()
    ):
        raise ValueError("invalid training data")
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-4)
    model = nn.Sequential(nn.Linear(x.shape[1], 64), nn.ReLU(), nn.Linear(64, 4))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    counts = np.bincount(y, minlength=4)
    weights = np.where(counts > 0, len(y) / (4 * np.maximum(counts, 1)), 0).astype(
        np.float32
    )
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights))
    train_x, train_y = torch.from_numpy((x - mean) / scale), torch.from_numpy(y)
    val_x, val_y = (
        torch.from_numpy((validation_x - mean) / scale),
        torch.from_numpy(validation_y),
    )
    history, best_accuracy, best = [], -1.0, None
    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(x))
        for batch in indices.split(256):
            optimizer.zero_grad()
            loss = loss_fn(model(train_x[batch]), train_y[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            accuracy = float((model(val_x).argmax(-1) == val_y).float().mean())
        history.append({"epoch": epoch, "validationAccuracy": accuracy})
        if accuracy > best_accuracy:
            best_accuracy, best = accuracy, copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    return Readout(mean, scale, model), history
