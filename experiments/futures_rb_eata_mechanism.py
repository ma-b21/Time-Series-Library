#!/usr/bin/env python3
import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader, Dataset

from layers.EATA_utils import EndogenousAnchoredTemporalAddressing


DATA_PATH = ROOT / "dataset/futures/futures_RB.csv"
OUTPUT_DIR = ROOT / "analysis_outputs/futures_rb_eata_mechanism"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FeatureScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, values: np.ndarray) -> None:
        self.mean = values.mean(axis=0, keepdims=True)
        self.std = values.std(axis=0, keepdims=True)
        self.std[self.std < 1e-6] = 1.0

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


class PairWindowDataset(Dataset):
    def __init__(self, values: np.ndarray, dates: np.ndarray, seq_len: int):
        self.values = values.astype(np.float32)
        self.dates = dates
        self.seq_len = seq_len

    def __len__(self):
        return len(self.values) - self.seq_len

    def __getitem__(self, index):
        seq = self.values[index : index + self.seq_len]
        target = self.values[index + self.seq_len, 1]
        return {
            "exo": torch.from_numpy(seq[:, 0:1]),
            "endo": torch.from_numpy(seq[:, 1:2]),
            "target": torch.tensor(target, dtype=torch.float32),
            "index": torch.tensor(index, dtype=torch.long),
        }

    def get_metadata(self, index):
        return {
            "window_dates": self.dates[index : index + self.seq_len],
            "pred_date": self.dates[index + self.seq_len],
        }


@dataclass
class SplitBundle:
    train: PairWindowDataset
    val: PairWindowDataset
    test: PairWindowDataset
    exo_scaler: FeatureScaler
    endo_scaler: FeatureScaler
    raw_frame: pd.DataFrame


def load_pair_dataset(exo_feature: str, target_feature: str, seq_len: int) -> SplitBundle:
    df = pd.read_csv(DATA_PATH)
    df = df[: len(df) // 2].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0.0)

    if exo_feature not in df.columns:
        raise ValueError(f"Unknown exogenous feature: {exo_feature}")
    if target_feature not in df.columns:
        raise ValueError(f"Unknown target feature: {target_feature}")

    frame = df[["date", exo_feature, target_feature]].copy()
    values = frame[[exo_feature, target_feature]].values.astype(np.float32)

    num_train = int(len(frame) * 0.7)
    num_test = int(len(frame) * 0.2)
    num_val = len(frame) - num_train - num_test
    border1s = [0, num_train - seq_len, len(frame) - num_test - seq_len]
    border2s = [num_train, num_train + num_val, len(frame)]

    train_slice = values[border1s[0] : border2s[0]]
    exo_scaler = FeatureScaler()
    endo_scaler = FeatureScaler()
    exo_scaler.fit(train_slice[:, 0:1])
    endo_scaler.fit(train_slice[:, 1:2])

    scaled = np.empty_like(values)
    scaled[:, 0:1] = exo_scaler.transform(values[:, 0:1])
    scaled[:, 1:2] = endo_scaler.transform(values[:, 1:2])

    dates = frame["date"].values
    train_ds = PairWindowDataset(scaled[border1s[0] : border2s[0]], dates[border1s[0] : border2s[0]], seq_len)
    val_ds = PairWindowDataset(scaled[border1s[1] : border2s[1]], dates[border1s[1] : border2s[1]], seq_len)
    test_ds = PairWindowDataset(scaled[border1s[2] : border2s[2]], dates[border1s[2] : border2s[2]], seq_len)
    return SplitBundle(train_ds, val_ds, test_ds, exo_scaler, endo_scaler, frame)


class EATASingleExoModel(nn.Module):
    def __init__(self, k_lookback: int, hidden_size: int, dropout: float):
        super().__init__()
        self.eata = EndogenousAnchoredTemporalAddressing(1, 1, k_lookback, dropout)
        self.temporal = nn.GRU(input_size=2, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, exo_seq, endo_seq, return_details=False):
        endo_c = endo_seq.transpose(1, 2)
        exo_c = exo_seq.transpose(1, 2)
        if return_details:
            combined, details = self.eata(endo_c, exo_c, return_components=True)
        else:
            combined = self.eata(endo_c, exo_c)
            details = None

        combined_seq = combined.squeeze(1).permute(0, 2, 1)
        temporal_out, _ = self.temporal(combined_seq)
        pred = self.head(temporal_out[:, -1]).squeeze(-1)

        if not return_details:
            return pred

        payload = {
            "exo_encoded": details["exo_encoded"].squeeze(1).squeeze(1),
            "signed_weights": details["signed_weights"].squeeze(1).squeeze(1),
            "weights": details["weights"].squeeze(1).squeeze(1),
            "sign_direction": details["sign_direction"].squeeze(1).squeeze(1),
            "product": details["product"].squeeze(1).squeeze(1),
            "key_value": details["key_value"].squeeze(1).squeeze(1),
            "combined_seq": combined_seq,
        }
        return pred, payload


class RNNSingleExoModel(nn.Module):
    def __init__(self, rnn_hidden: int, hidden_size: int):
        super().__init__()
        self.exo_rnn = nn.RNN(input_size=1, hidden_size=rnn_hidden, batch_first=True)
        self.hidden_to_scalar = nn.Linear(rnn_hidden, 1)
        self.temporal = nn.GRU(input_size=2, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, exo_seq, endo_seq, return_details=False):
        hidden_seq, _ = self.exo_rnn(exo_seq)
        hidden_scalar = self.hidden_to_scalar(hidden_seq)
        combined = torch.cat([hidden_scalar, endo_seq], dim=-1)
        temporal_out, _ = self.temporal(combined)
        pred = self.head(temporal_out[:, -1]).squeeze(-1)
        if not return_details:
            return pred
        return pred, {"hidden_states": hidden_scalar.squeeze(-1), "raw_hidden": hidden_seq}


def make_loader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.MSELoss()
    losses = []
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            exo = batch["exo"].to(device)
            endo = batch["endo"].to(device)
            target = batch["target"].to(device)
            pred = model(exo, endo)
            losses.append(loss_fn(pred, target).item())
            preds.append(pred.cpu().numpy())
            trues.append(target.cpu().numpy())
    preds = np.concatenate(preds) if preds else np.array([])
    trues = np.concatenate(trues) if trues else np.array([])
    mae = float(np.mean(np.abs(preds - trues))) if len(preds) else math.inf
    mse = float(np.mean((preds - trues) ** 2)) if len(preds) else math.inf
    return {"loss": float(np.mean(losses)) if losses else math.inf, "mae": mae, "mse": mse}


def train_model(model, train_loader, val_loader, device, epochs, lr, patience):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_state = None
    best_metric = math.inf
    wait = 0

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            exo = batch["exo"].to(device)
            endo = batch["endo"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(exo, endo)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate(model, val_loader, device)
        if val_metrics["mae"] < best_metric:
            best_metric = val_metrics["mae"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def collect_predictions(model, dataset, device):
    loader = make_loader(dataset, batch_size=256, shuffle=False)
    model.eval()
    preds, trues, endo_last = [], [], []
    with torch.no_grad():
        for batch in loader:
            exo = batch["exo"].to(device)
            endo = batch["endo"].to(device)
            target = batch["target"].to(device)
            pred = model(exo, endo)
            preds.append(pred.cpu().numpy())
            trues.append(target.cpu().numpy())
            endo_last.append(endo[:, -1, 0].cpu().numpy())
    return (
        np.concatenate(preds),
        np.concatenate(trues),
        np.concatenate(endo_last),
    )


def choose_showcase_index(eata_preds, rnn_preds, trues, endo_last):
    eata_err = np.abs(eata_preds - trues)
    rnn_err = np.abs(rnn_preds - trues)
    move = np.abs(trues - endo_last)

    move_threshold = np.quantile(move, 0.75)
    good_err_threshold = np.quantile(eata_err, 0.25)
    candidate = np.where(
        (move >= move_threshold)
        & (eata_err <= good_err_threshold)
        & (eata_err <= rnn_err + 1e-8)
    )[0]
    if len(candidate) == 0:
        candidate = np.where(eata_err <= np.quantile(eata_err, 0.1))[0]
    if len(candidate) == 0:
        return int(np.argmin(eata_err))

    score = (rnn_err[candidate] - eata_err[candidate]) + 0.1 * move[candidate]
    return int(candidate[np.argmax(score)])


def batch_to_device(sample, device):
    exo = sample["exo"].unsqueeze(0).to(device)
    endo = sample["endo"].unsqueeze(0).to(device)
    return exo, endo


def style_axis(ax, grid_axis="both"):
    ax.grid(axis=grid_axis, alpha=0.32, linestyle="-", linewidth=0.75, color="#b8bec7")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)


def draw_figure_one(ax, exo_series, signed_weights_last):
    x = np.arange(1, len(exo_series) + 1)
    signed_weights_last = np.asarray(signed_weights_last, dtype=float)
    weight_min = float(np.min(signed_weights_last))
    weight_max = float(np.max(signed_weights_last))
    if abs(weight_max - weight_min) < 1e-12:
        weight_max = weight_min + 1e-12
    norm = Normalize(vmin=weight_min, vmax=weight_max)

    ax.plot(x, exo_series, color="#2f3640", linewidth=1.6, alpha=0.9)
    scatter = ax.scatter(
        x,
        exo_series,
        c=signed_weights_last,
        cmap="coolwarm",
        norm=norm,
        s=64,
        edgecolors="white",
        linewidths=0.45,
        zorder=3,
    )
    ax.set_xlim(1, len(exo_series))
    ax.set_xlabel("Time Index Within Window")
    ax.set_ylabel("Exogenous variable (z-score)")
    style_axis(ax)
    return scatter


def plot_figure_one(exo_series, signed_weights_last, exo_name, out_path):
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    scatter = draw_figure_one(ax, exo_series, signed_weights_last)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("signed weight at t=96")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def draw_figure_two(ax, exo_encoded, rnn_hidden):
    x = np.arange(1, len(exo_encoded) + 1)
    ax.plot(x, exo_encoded, color="#7A6FD6", linewidth=2.0, label="EATA exo_encoded")
    ax.plot(
        x,
        rnn_hidden,
        color="#67B7D1",
        linewidth=1.8,
        linestyle=(0, (5, 3)),
        label="RNN hidden states",
    )
    ax.set_xlim(1, len(exo_encoded))
    ax.set_xlabel("Time Index Within Window")
    ax.set_ylabel("Encoded Value (z-score)")
    style_axis(ax)
    ax.legend(loc="upper right", fontsize=8.8, frameon=False)


def plot_figure_two(exo_encoded, rnn_hidden, out_path):
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    draw_figure_two(ax, exo_encoded, rnn_hidden)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def draw_figure_three(ax, eata_pred, rnn_pred, true_value):
    labels = ["True", "EATA", "RNN"]
    values = np.array([true_value, eata_pred, rnn_pred], dtype=float)
    colors = ["#FFE44D", "#7A6FD6", "#67B7D1"]
    markers = ["*", "o", "^"]
    sizes = [260, 110, 120]
    x = np.arange(len(labels))

    value_span = float(values.max() - values.min())
    margin = max(value_span * 0.35, 0.6)
    text_offset = max(value_span * 0.08, 0.12)

    ax.axhline(true_value, color="#9aa0a6", linestyle="--", linewidth=1.0, alpha=0.85)
    ax.vlines([1, 2], true_value, [eata_pred, rnn_pred], colors=[colors[1], colors[2]], linewidth=2.0, alpha=0.85)

    for idx, (value, color, marker, size) in enumerate(zip(values, colors, markers, sizes)):
        ax.scatter(
            [x[idx]],
            [value],
            color=color,
            marker=marker,
            s=size,
            edgecolors="black",
            linewidths=0.45,
            zorder=4,
        )
        ax.text(x[idx], value + text_offset, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    ax.text(1, eata_pred - text_offset * 1.2, f"|err|={abs(eata_pred - true_value):.3f}", ha="center", va="top", fontsize=8.5, color=colors[1])
    ax.text(2, rnn_pred - text_offset * 1.2, f"|err|={abs(rnn_pred - true_value):.3f}", ha="center", va="top", fontsize=8.5, color=colors[2])

    ax.set_xlim(-0.4, 2.4)
    ax.set_ylim(values.min() - margin, values.max() + margin)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Target (endogenous variable)")
    style_axis(ax, grid_axis="y")


def plot_figure_three(eata_pred, rnn_pred, true_value, target_name, out_path):
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    draw_figure_three(ax, eata_pred, rnn_pred, true_value)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_combined_figure(exo_series, signed_weights_last, exo_encoded, rnn_hidden, eata_pred, rnn_pred, true_value, out_path):
    fig = plt.figure(figsize=(16.8, 7.2), constrained_layout=True)
    grid = GridSpec(2, 2, figure=fig, width_ratios=[2.25, 1.05], wspace=0.20, hspace=0.22)

    top_left = grid[0, 0].subgridspec(1, 2, width_ratios=[24, 1.3], wspace=0.05)
    ax1 = fig.add_subplot(top_left[0, 0])
    scatter = draw_figure_one(ax1, exo_series, signed_weights_last)
    cax = fig.add_subplot(top_left[0, 1])
    cbar = fig.colorbar(scatter, cax=cax)
    cbar.set_label("signed weight at t=96")

    ax2 = fig.add_subplot(grid[1, 0])
    draw_figure_two(ax2, exo_encoded, rnn_hidden)

    right_grid = grid[:, 1].subgridspec(3, 1, height_ratios=[0.22, 1.0, 0.22], hspace=0.0)
    ax3 = fig.add_subplot(right_grid[1, 0])
    ax3.set_box_aspect(1)
    draw_figure_three(ax3, eata_pred, rnn_pred, true_value)

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_window_csv(out_path, window_dates, exo_series, endo_series, exo_encoded, rnn_hidden):
    df = pd.DataFrame(
        {
            "date": window_dates,
            "exo_zscore": exo_series,
            "endo_zscore": endo_series,
            "exo_encoded_zscore": exo_encoded,
            "rnn_hidden_zscore": rnn_hidden,
        }
    )
    df.to_csv(out_path, index=False)


def save_signed_weight_csv(out_path, window_dates, exo_series, signed_weights_last):
    df = pd.DataFrame(
        {
            "lag_index": np.arange(1, len(signed_weights_last) + 1),
            "date": window_dates,
            "exo_zscore": exo_series,
            "signed_weight": signed_weights_last,
        }
    )
    df.to_csv(out_path, index=False)


def replot_from_saved_outputs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "summary.json"
    series_path = OUTPUT_DIR / "selected_window_series.csv"
    weights_path = OUTPUT_DIR / "selected_t96_signed_weights.csv"

    summary = json.loads(summary_path.read_text())
    series_df = pd.read_csv(series_path)
    weights_df = pd.read_csv(weights_path)

    fig1_path = OUTPUT_DIR / "figure1_eata_weighted_exo.png"
    fig2_path = OUTPUT_DIR / "figure2_eata_vs_rnn.png"
    fig3_path = OUTPUT_DIR / "figure3_target_prediction_comparison.png"
    plot_figure_one(
        series_df["exo_zscore"].to_numpy(),
        weights_df["signed_weight"].to_numpy(),
        summary["exogenous_feature"],
        fig1_path,
    )
    plot_figure_two(
        series_df["exo_encoded_zscore"].to_numpy(),
        series_df["rnn_hidden_zscore"].to_numpy(),
        fig2_path,
    )
    plot_figure_three(
        summary["eata_pred_original_scale"],
        summary["rnn_pred_original_scale"],
        summary["true_value_original_scale"],
        summary["target_feature"],
        fig3_path,
    )

    summary["figure1_path"] = str(fig1_path.relative_to(ROOT))
    summary["figure2_path"] = str(fig2_path.relative_to(ROOT))
    summary["figure3_path"] = str(fig3_path.relative_to(ROOT))
    summary.pop("figure_combined_path", None)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def run_experiment(args):
    set_seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bundle = load_pair_dataset(args.exo_feature, args.target_feature, args.seq_len)
    train_loader = make_loader(bundle.train, args.batch_size, shuffle=True)
    val_loader = make_loader(bundle.val, args.batch_size, shuffle=False)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    eata_model = EATASingleExoModel(args.k_lookback, args.hidden_size, args.dropout).to(device)
    rnn_model = RNNSingleExoModel(args.rnn_hidden, args.hidden_size).to(device)

    eata_model = train_model(eata_model, train_loader, val_loader, device, args.epochs, args.learning_rate, args.patience)
    rnn_model = train_model(rnn_model, train_loader, val_loader, device, args.epochs, args.learning_rate, args.patience)

    eata_preds, trues, endo_last = collect_predictions(eata_model, bundle.test, device)
    rnn_preds, _, _ = collect_predictions(rnn_model, bundle.test, device)
    showcase_idx = choose_showcase_index(eata_preds, rnn_preds, trues, endo_last)

    sample = bundle.test[showcase_idx]
    sample_meta = bundle.test.get_metadata(showcase_idx)
    exo, endo = batch_to_device(sample, device)

    eata_model.eval()
    rnn_model.eval()
    with torch.no_grad():
        eata_pred, eata_details = eata_model(exo, endo, return_details=True)
        rnn_pred, rnn_details = rnn_model(exo, endo, return_details=True)

    exo_series = sample["exo"][:, 0].numpy()
    endo_series = sample["endo"][:, 0].numpy()
    exo_encoded = eata_details["exo_encoded"][0].cpu().numpy()
    signed_weights_last = eata_details["signed_weights"][0, -1].cpu().numpy()
    rnn_hidden = rnn_details["hidden_states"][0].cpu().numpy()
    true_value = float(sample["target"].item())
    eata_pred_value = float(eata_pred.item())
    rnn_pred_value = float(rnn_pred.item())

    save_window_csv(OUTPUT_DIR / "selected_window_series.csv", sample_meta["window_dates"], exo_series, endo_series, exo_encoded, rnn_hidden)
    save_signed_weight_csv(OUTPUT_DIR / "selected_t96_signed_weights.csv", sample_meta["window_dates"], exo_series, signed_weights_last)

    target_scaler = bundle.endo_scaler
    eata_pred_orig = float(target_scaler.inverse_transform(np.array([[eata_pred_value]], dtype=np.float32))[0, 0])
    rnn_pred_orig = float(target_scaler.inverse_transform(np.array([[rnn_pred_value]], dtype=np.float32))[0, 0])
    true_orig = float(target_scaler.inverse_transform(np.array([[true_value]], dtype=np.float32))[0, 0])

    fig1_path = OUTPUT_DIR / "figure1_eata_weighted_exo.png"
    fig2_path = OUTPUT_DIR / "figure2_eata_vs_rnn.png"
    fig3_path = OUTPUT_DIR / "figure3_target_prediction_comparison.png"
    plot_figure_one(exo_series, signed_weights_last, args.exo_feature, fig1_path)
    plot_figure_two(exo_encoded, rnn_hidden, fig2_path)
    plot_figure_three(eata_pred_orig, rnn_pred_orig, true_orig, args.target_feature, fig3_path)

    summary = {
        "dataset": "Futures_RB",
        "target_feature": args.target_feature,
        "exogenous_feature": args.exo_feature,
        "selected_window_index_in_test": showcase_idx,
        "window_start_date": str(sample_meta["window_dates"][0]),
        "window_end_date": str(sample_meta["window_dates"][-1]),
        "prediction_date_t97": str(sample_meta["pred_date"]),
        "eata_pred_zscore": eata_pred_value,
        "rnn_pred_zscore": rnn_pred_value,
        "true_value_zscore": true_value,
        "eata_pred_original_scale": eata_pred_orig,
        "rnn_pred_original_scale": rnn_pred_orig,
        "true_value_original_scale": true_orig,
        "test_mae_eata_zscore": float(np.mean(np.abs(eata_preds - trues))),
        "test_mae_rnn_zscore": float(np.mean(np.abs(rnn_preds - trues))),
        "figure1_path": str(fig1_path.relative_to(ROOT)),
        "figure2_path": str(fig2_path.relative_to(ROOT)),
        "figure3_path": str(fig3_path.relative_to(ROOT)),
        "series_csv_path": str((OUTPUT_DIR / "selected_window_series.csv").relative_to(ROOT)),
        "weights_csv_path": str((OUTPUT_DIR / "selected_t96_signed_weights.csv").relative_to(ROOT)),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exo-feature", default="LastPrice")
    parser.add_argument("--target-feature", default="ClosePrice")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--k-lookback", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--rnn-hidden", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="")
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.plot_only:
        replot_from_saved_outputs()
    else:
        run_experiment(args)
