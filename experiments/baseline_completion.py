#!/usr/bin/env python3
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN_PY = ROOT / "run.py"
RESULTS_DIR = ROOT / "results"
TABLES_DIR = Path("/root/autodl-tmp/tables")
ALL_RESULTS_PATH = TABLES_DIR / "allResults.tex"
SINGLE_FULL_PATH = TABLES_DIR / "singleFull.tex"

REPORT_PATH = ROOT / "baseline_progress_summary.md"
RUNS_PATH = ROOT / "baseline_completion_runs.jsonl"
STATE_PATH = ROOT / "baseline_completion_state.json"
LOG_DIR = ROOT / "baseline_completion_logs"

MULTI_EXISTING_MODELS = [
    "Ours",
    "TimeXer",
    "TimeMixer",
    "iTransformer",
    "PatchTST",
    "Autoformer",
]
SINGLE_EXISTING_MODELS = ["Ours", "TimeXer"]
NEW_MODELS = ["Crossformer", "DLinear", "TimesNet"]
DEFAULT_SCHEDULE_MODELS = ["DLinear", "Crossformer"]

MULTI_DATASET_ORDER = [
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "Exchange",
    "ILI",
    "Weather",
    "Futures_RB",
    "Futures_TA",
    "Futures_M",
]
SINGLE_DATASET_ORDER = [
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "Weather",
    "Exchange",
    "ILI",
    "Futures_TA",
    "Futures_RB",
    "Futures_M",
]

LATEX_DATASET = {
    "ETTh1": r"\textbf{ETTh1}",
    "ETTh2": r"\textbf{ETTh2}",
    "ETTm1": r"\textbf{ETTm1}",
    "ETTm2": r"\textbf{ETTm2}",
    "Exchange": r"\textbf{Exchange}",
    "ILI": r"\textbf{ILI}",
    "Weather": r"\textbf{Weather}",
    "Futures_RB": r"\textbf{Futures\_RB}",
    "Futures_TA": r"\textbf{Futures\_TA}",
    "Futures_M": r"\textbf{Futures\_M}",
}

HORIZONS = {
    "ETTh1": [96, 192, 336, 720],
    "ETTh2": [96, 192, 336, 720],
    "ETTm1": [96, 192, 336, 720],
    "ETTm2": [96, 192, 336, 720],
    "Exchange": [96, 192, 336, 720],
    "ILI": [24, 36, 48, 60],
    "Weather": [96, 192, 336, 720],
    "Futures_RB": [96, 192, 336, 720],
    "Futures_TA": [96, 192, 336, 720],
    "Futures_M": [96, 192, 336, 720],
}

DATASET_CONFIGS = {
    "ETTh1": {
        "data": "ETTh1",
        "root_path": "./dataset/ETT-small/",
        "data_path": "ETTh1.csv",
        "target": "OT",
        "seq_len": 96,
        "label_len": 48,
        "enc_in": 7,
    },
    "ETTh2": {
        "data": "ETTh2",
        "root_path": "./dataset/ETT-small/",
        "data_path": "ETTh2.csv",
        "target": "OT",
        "seq_len": 96,
        "label_len": 48,
        "enc_in": 7,
    },
    "ETTm1": {
        "data": "ETTm1",
        "root_path": "./dataset/ETT-small/",
        "data_path": "ETTm1.csv",
        "target": "OT",
        "seq_len": 96,
        "label_len": 48,
        "enc_in": 7,
    },
    "ETTm2": {
        "data": "ETTm2",
        "root_path": "./dataset/ETT-small/",
        "data_path": "ETTm2.csv",
        "target": "OT",
        "seq_len": 96,
        "label_len": 48,
        "enc_in": 7,
    },
    "Exchange": {
        "data": "custom",
        "root_path": "./dataset/exchange_rate/",
        "data_path": "exchange_rate.csv",
        "target": "OT",
        "seq_len": 96,
        "label_len": 48,
        "enc_in": 8,
    },
    "ILI": {
        "data": "custom",
        "root_path": "./dataset/illness/",
        "data_path": "national_illness.csv",
        "target": "OT",
        "seq_len": 36,
        "label_len": 18,
        "enc_in": 7,
    },
    "Weather": {
        "data": "custom",
        "root_path": "./dataset/weather/",
        "data_path": "weather.csv",
        "target": "OT",
        "seq_len": 96,
        "label_len": 48,
        "enc_in": 21,
    },
    "Futures_RB": {
        "data": "Futures",
        "root_path": "./dataset/futures/",
        "data_path": "futures_RB.csv",
        "target": "LastPrice",
        "seq_len": 96,
        "label_len": 48,
        "enc_in_m": 12,
        "enc_in_ms": 146,
    },
    "Futures_TA": {
        "data": "Futures",
        "root_path": "./dataset/futures/",
        "data_path": "futures_TA.csv",
        "target": "LastPrice",
        "seq_len": 96,
        "label_len": 48,
        "enc_in_m": 12,
        "enc_in_ms": 146,
    },
    "Futures_M": {
        "data": "Futures",
        "root_path": "./dataset/futures/",
        "data_path": "futures_M.csv",
        "target": "LastPrice",
        "seq_len": 96,
        "label_len": 48,
        "enc_in_m": 12,
        "enc_in_ms": 146,
    },
}

RUN_DEFAULTS = {
    "task_name": "long_term_forecast",
    "is_training": "1",
    "num_workers": "1",
    "dropout": "0.1",
    "batch_size": "32",
    "learning_rate": "0.0001",
    "patience": "3",
    "train_epochs": "10",
    "itr": "1",
    "d_model": "512",
    "d_ff": "2048",
    "e_layers": "2",
    "d_layers": "1",
    "factor": "3",
    "k_lookback": "64",
    "hidden": "10",
    "des": "BaseComp",
}

MODEL_PRIORITY = {"DLinear": 0, "Crossformer": 1, "TimesNet": 2}

METRIC_FILE_RE = re.compile(r"\\(?:best|second)\{([^}]*)\}")


def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def strip_metric_cell(cell):
    cleaned = cell.strip()
    cleaned = METRIC_FILE_RE.sub(r"\1", cleaned)
    cleaned = cleaned.replace(r"\textbf{", "").replace("}", "")
    return cleaned


def parse_metric_value(cell):
    cleaned = strip_metric_cell(cell).strip()
    if cleaned in {"", "-"}:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    return float(match.group(0))


def metric_pair_from_cells(mse_cell, mae_cell):
    mse = parse_metric_value(mse_cell)
    mae = parse_metric_value(mae_cell)
    if mse is None or mae is None:
        return None
    return {"mse": mse, "mae": mae}


def metric_or_dash(metrics, key):
    if metrics is None:
        return "-"
    return f"{metrics[key]:.3f}"


def average_metrics(metric_list):
    if not metric_list or any(item is None for item in metric_list):
        return None
    return {
        "mse": float(sum(item["mse"] for item in metric_list) / len(metric_list)),
        "mae": float(sum(item["mae"] for item in metric_list) / len(metric_list)),
    }


def build_jobs(selected_models=None, num_workers=None):
    selected_models = selected_models or DEFAULT_SCHEDULE_MODELS
    jobs = []
    for mode in ("M", "MS"):
        for dataset in MULTI_DATASET_ORDER:
            for pred_len in HORIZONS[dataset]:
                for model in selected_models:
                    job = make_job(mode, dataset, pred_len, model)
                    if num_workers is not None:
                        job["args"]["num_workers"] = str(num_workers)
                    jobs.append(job)
    return sorted(jobs, key=job_sort_key)


def make_job(mode, dataset, pred_len, model):
    cfg = deepcopy(DATASET_CONFIGS[dataset])
    seq_len = cfg["seq_len"]
    label_len = cfg["label_len"]
    if dataset.startswith("Futures_"):
        enc_in = cfg["enc_in_m"] if mode == "M" else cfg["enc_in_ms"]
    else:
        enc_in = cfg["enc_in"]
    args = {
        "task_name": "long_term_forecast",
        "is_training": "1",
        "model": model,
        "model_id": f"{dataset}_{mode}_{seq_len}_{pred_len}",
        "data": cfg["data"],
        "root_path": cfg["root_path"],
        "data_path": cfg["data_path"],
        "target": cfg["target"],
        "features": mode,
        "seq_len": str(seq_len),
        "label_len": str(label_len),
        "pred_len": str(pred_len),
        "enc_in": str(enc_in),
        "dec_in": str(enc_in),
        "c_out": str(enc_in),
        "des": f"BaseComp-{mode}",
        "num_workers": RUN_DEFAULTS["num_workers"],
        "itr": RUN_DEFAULTS["itr"],
    }
    args.update(model_overrides(model, dataset, mode, pred_len))
    return {
        "mode": mode,
        "dataset": dataset,
        "pred_len": pred_len,
        "model": model,
        "args": args,
    }


def job_key(job):
    return (job["mode"], job["dataset"], job["pred_len"], job["model"])


def model_overrides(model, dataset, mode, pred_len):
    overrides = {
        "dropout": RUN_DEFAULTS["dropout"],
        "batch_size": RUN_DEFAULTS["batch_size"],
        "learning_rate": RUN_DEFAULTS["learning_rate"],
        "patience": RUN_DEFAULTS["patience"],
        "train_epochs": RUN_DEFAULTS["train_epochs"],
        "d_model": RUN_DEFAULTS["d_model"],
        "d_ff": RUN_DEFAULTS["d_ff"],
        "e_layers": RUN_DEFAULTS["e_layers"],
        "d_layers": RUN_DEFAULTS["d_layers"],
        "factor": RUN_DEFAULTS["factor"],
    }
    if model == "DLinear":
        if dataset == "ILI":
            overrides["batch_size"] = "16"
        elif dataset.startswith("Futures_") and mode == "MS":
            overrides["batch_size"] = "16"
        return overrides

    if model == "TimesNet":
        overrides["top_k"] = "5"
        if dataset.startswith("ETT"):
            overrides["d_model"] = "16"
            overrides["d_ff"] = "32"
        elif dataset == "Exchange":
            dim = "64" if pred_len in (96, 192) else "32"
            overrides["d_model"] = dim
            overrides["d_ff"] = dim
        elif dataset == "Weather":
            overrides["d_model"] = "32"
            overrides["d_ff"] = "32"
        elif dataset == "ILI":
            overrides["d_model"] = "768"
            overrides["d_ff"] = "768"
            overrides["batch_size"] = "16" if mode == "MS" else "32"
        elif dataset.startswith("Futures_"):
            overrides["d_model"] = "32"
            overrides["d_ff"] = "64"
            overrides["batch_size"] = "4" if mode == "MS" else "16"
        if mode == "MS" and dataset in {"Weather", "Exchange"}:
            overrides["batch_size"] = "16"
        return overrides

    if model == "Crossformer":
        if dataset == "Exchange":
            dim = "64" if pred_len in (96, 192) else "32"
            overrides["d_model"] = dim
            overrides["d_ff"] = dim
        elif dataset == "Weather":
            overrides["d_model"] = "32"
            overrides["d_ff"] = "32"
        elif dataset == "ILI":
            overrides["d_model"] = "768"
            overrides["d_ff"] = "768"
            overrides["dropout"] = "0.6"
            overrides["batch_size"] = "16"
        elif dataset.startswith("Futures_"):
            overrides["d_model"] = "64"
            overrides["d_ff"] = "64"
            overrides["batch_size"] = "4" if mode == "MS" else "16"
        if mode == "MS" and dataset in {"Weather", "Exchange"}:
            overrides["batch_size"] = "8"
        return overrides

    return overrides


def job_sort_key(job):
    dataset = job["dataset"]
    pred_len = job["pred_len"]
    model_rank = MODEL_PRIORITY[job["model"]]
    if dataset == "ILI":
        horizon_rank = {24: 0, 36: 1, 48: 2, 60: 3}[pred_len]
    else:
        horizon_rank = {96: 0, 192: 1, 336: 2, 720: 3}[pred_len]
    mode_rank = 0 if job["mode"] == "M" else 1
    futures_rank = 1 if dataset.startswith("Futures_") else 0
    return (model_rank, futures_rank, mode_rank, dataset, horizon_rank)


def setting_from_args(args):
    merged = deepcopy(RUN_DEFAULTS)
    merged.update({k: str(v) for k, v in args.items() if v is not True})
    return (
        f"{merged['task_name']}_{merged['model_id']}_{merged['model']}_{merged['data']}"
        f"_bs{merged['batch_size']}_lr{merged['learning_rate']}_pt{merged['patience']}"
        f"_dp{merged['dropout']}_dm{merged['d_model']}_lb{merged['k_lookback']}"
        f"_hidden{merged['hidden']}_{merged['des']}"
    )


def result_metrics(setting):
    metrics_path = RESULTS_DIR / setting / "metrics.npy"
    if not metrics_path.exists():
        return None
    mae, mse, rmse, mape, mspe = np.load(metrics_path)
    return {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "mspe": float(mspe),
    }


def job_to_command(args, gpu):
    ordered_keys = [
        "task_name",
        "is_training",
        "model_id",
        "model",
        "data",
        "root_path",
        "data_path",
        "target",
        "features",
        "seq_len",
        "label_len",
        "pred_len",
        "enc_in",
        "dec_in",
        "c_out",
        "e_layers",
        "d_layers",
        "factor",
        "d_model",
        "d_ff",
        "dropout",
        "batch_size",
        "learning_rate",
        "train_epochs",
        "patience",
        "num_workers",
        "des",
        "top_k",
        "itr",
        "gpu",
    ]
    merged = deepcopy(args)
    merged["gpu"] = str(gpu)
    command = ["python", "run.py"]
    for key in ordered_keys:
        if key not in merged:
            continue
        value = merged[key]
        if value is True:
            command.append(f"--{key}")
        else:
            command.extend([f"--{key}", str(value)])
    for key, value in merged.items():
        if key in ordered_keys:
            continue
        if value is True:
            command.append(f"--{key}")
        else:
            command.extend([f"--{key}", str(value)])
    return command


def append_run_record(record):
    with RUNS_PATH.open("a") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def load_run_records():
    records = []
    if not RUNS_PATH.exists():
        return records
    for line in RUNS_PATH.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def record_completion(job, status, returncode, log_path, metrics=None):
    append_run_record(
        {
            "finished_at": now_utc(),
            "status": status,
            "returncode": returncode,
            "setting": setting_from_args(job["args"]),
            "model": job["model"],
            "mode": job["mode"],
            "dataset": job["dataset"],
            "pred_len": job["pred_len"],
            "metrics": metrics,
            "log_path": str(log_path.relative_to(ROOT)) if log_path else "",
            "args": job["args"],
        }
    )


def completed_metrics_by_job(jobs):
    completed = existing_table_metrics()
    for job in jobs:
        setting = setting_from_args(job["args"])
        metrics = result_metrics(setting)
        if metrics is not None:
            completed[job_key(job)] = metrics
    return completed


def progress_counts(jobs, completed):
    totals = defaultdict(int)
    done = defaultdict(int)
    for job in jobs:
        key = (job["mode"], job["model"])
        totals[key] += 1
        if job_key(job) in completed:
            done[key] += 1
    return totals, done


def scheduled_models(jobs):
    return sorted({job["model"] for job in jobs}, key=lambda model: MODEL_PRIORITY.get(model, 999))


def write_state(jobs, pending, running):
    completed = completed_metrics_by_job(jobs)
    totals, done = progress_counts(jobs, completed)
    models = scheduled_models(jobs)
    payload = {
        "updated_at": now_utc(),
        "scheduler_pid": os.getpid(),
        "total_jobs": len(jobs),
        "completed_jobs": len(completed),
        "pending_jobs": len(pending),
        "running_jobs": [
            {
                "gpu": gpu,
                "pid": item["proc"].pid,
                "mode": item["job"]["mode"],
                "dataset": item["job"]["dataset"],
                "pred_len": item["job"]["pred_len"],
                "model": item["job"]["model"],
                "setting": item["setting"],
                "log_path": str(item["log_path"].relative_to(ROOT)),
                "launched_at": item["launched_at"],
            }
            for gpu, item in sorted(running.items())
        ],
        "progress_by_mode_model": {
            f"{mode}:{model}": {"completed": done[(mode, model)], "total": totals[(mode, model)]}
            for mode in ("M", "MS")
            for model in models
        },
    }
    STATE_PATH.write_text(json.dumps(payload, indent=2))


def format_job_matrix_lines():
    lines = []
    lines.append("## Experiment Inventory")
    lines.append("")
    lines.append("### Multivariate (`M`)")
    lines.append("")
    for dataset in MULTI_DATASET_ORDER:
        horizon_str = ", ".join(str(item) for item in HORIZONS[dataset])
        lines.append(f"- `{dataset}`: {horizon_str}")
    lines.append("")
    lines.append("### Single-Target (`MS`)")
    lines.append("")
    for dataset in SINGLE_DATASET_ORDER:
        horizon_str = ", ".join(str(item) for item in HORIZONS[dataset])
        lines.append(f"- `{dataset}`: {horizon_str}")
    lines.append("")
    return lines


def write_report(jobs, pending, running):
    completed = completed_metrics_by_job(jobs)
    totals, done = progress_counts(jobs, completed)
    records = load_run_records()
    models = scheduled_models(jobs)

    lines = ["# Baseline Completion Report", ""]
    lines.append(f"Updated: {now_utc()}")
    lines.append(f"Scheduler PID: `{os.getpid()}`")
    lines.append(f"Total jobs: `{len(jobs)}`")
    lines.append(f"Completed jobs: `{len(completed)}`")
    lines.append(f"Pending jobs: `{len(pending)}`")
    lines.append(f"Running jobs: `{len(running)}`")
    lines.append("")

    lines.extend(format_job_matrix_lines())

    lines.append("## Progress Summary")
    lines.append("")
    lines.append("| Mode | Model | Completed | Total |")
    lines.append("| --- | --- | ---: | ---: |")
    for mode in ("M", "MS"):
        for model in models:
            lines.append(f"| {mode} | {model} | {done[(mode, model)]} | {totals[(mode, model)]} |")
    lines.append("")

    lines.append("## Multivariate Results (`M`)")
    lines.append("")
    lines.append("| Dataset | Horizon | Crossformer MSE | Crossformer MAE | DLinear MSE | DLinear MAE | TimesNet MSE | TimesNet MAE |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for dataset in MULTI_DATASET_ORDER:
        for pred_len in HORIZONS[dataset]:
            row = [dataset, str(pred_len)]
            for model in NEW_MODELS:
                metrics = completed.get(("M", dataset, pred_len, model))
                row.extend([metric_or_dash(metrics, "mse"), metric_or_dash(metrics, "mae")])
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Single-Target Results (`MS`)")
    lines.append("")
    lines.append("| Dataset | Horizon | Crossformer MSE | Crossformer MAE | DLinear MSE | DLinear MAE | TimesNet MSE | TimesNet MAE |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for dataset in SINGLE_DATASET_ORDER:
        for pred_len in HORIZONS[dataset]:
            row = [dataset, str(pred_len)]
            for model in NEW_MODELS:
                metrics = completed.get(("MS", dataset, pred_len, model))
                row.extend([metric_or_dash(metrics, "mse"), metric_or_dash(metrics, "mae")])
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    if running:
        lines.append("## Running Jobs")
        lines.append("")
        lines.append("| GPU | PID | Mode | Dataset | Horizon | Model | Log |")
        lines.append("| --- | ---: | --- | --- | ---: | --- | --- |")
        for gpu, item in sorted(running.items()):
            lines.append(
                f"| {gpu} | {item['proc'].pid} | {item['job']['mode']} | {item['job']['dataset']} | "
                f"{item['job']['pred_len']} | {item['job']['model']} | `{item['log_path'].relative_to(ROOT)}` |"
            )
        lines.append("")

    if records:
        lines.append("## Recent Finished Runs")
        lines.append("")
        lines.append("| Time | Status | Mode | Dataset | Horizon | Model | MSE | MAE | Log |")
        lines.append("| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- |")
        for record in reversed(records[-30:]):
            metrics = record.get("metrics") or {}
            mse = "-" if "mse" not in metrics else f"{metrics['mse']:.3f}"
            mae = "-" if "mae" not in metrics else f"{metrics['mae']:.3f}"
            lines.append(
                f"| {record.get('finished_at', '-')} | {record.get('status', '-')} | {record.get('mode', '-')} | "
                f"{record.get('dataset', '-')} | {record.get('pred_len', '-')} | {record.get('model', '-')} | "
                f"{mse} | {mae} | `{record.get('log_path', '-')}` |"
            )
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines))


def parse_multivariate_existing():
    text = ALL_RESULTS_PATH.read_text()
    rows = {}
    current_dataset = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if (
            not line
            or line.startswith("%")
            or "\\midrule" in line
            or "\\bottomrule" in line
            or "\\multicolumn" in line
            or "\\cmidrule" in line
            or "MSE & MAE" in line
            or "& & MSE" in line
        ):
            continue
        if "\\multirow" in line:
            dataset_match = re.search(r"\\textbf\{([^}]*)\}", line)
            if dataset_match:
                current_dataset = dataset_match.group(1).replace(r"\_", "_")
        if "\\multirow" not in line and not line.startswith("&"):
            continue
        if "&" not in line:
            continue
        parts = [part.strip() for part in line.rstrip("\\").split("&")]
        if len(parts) < 3:
            continue
        if current_dataset is None:
            continue
        horizon = parts[1].replace(r"\textbf{Avg}", "Avg").strip()
        if horizon == "Avg":
            continue
        rows[(current_dataset, int(horizon))] = parts[2:14]
    return rows


def parse_multivariate_new_model_metrics():
    if not ALL_RESULTS_PATH.exists():
        return {}
    text = ALL_RESULTS_PATH.read_text()
    rows = {}
    current_dataset = None
    model_offsets = {"Crossformer": 14, "DLinear": 16, "TimesNet": 18}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if (
            not line
            or line.startswith("%")
            or "\\midrule" in line
            or "\\bottomrule" in line
            or "\\multicolumn" in line
            or "\\cmidrule" in line
            or "MSE & MAE" in line
            or "& & MSE" in line
        ):
            continue
        if "\\multirow" in line:
            dataset_match = re.search(r"\\textbf\{([^}]*)\}", line)
            if dataset_match:
                current_dataset = dataset_match.group(1).replace(r"\_", "_")
        if "\\multirow" not in line and not line.startswith("&"):
            continue
        if "&" not in line or current_dataset is None:
            continue
        parts = [part.strip() for part in line.rstrip("\\").split("&")]
        if len(parts) < 20:
            continue
        horizon = parts[1].replace(r"\textbf{Avg}", "Avg").strip()
        if horizon == "Avg":
            continue
        pred_len = int(horizon)
        for model, offset in model_offsets.items():
            metrics = metric_pair_from_cells(parts[offset], parts[offset + 1])
            if metrics is not None:
                rows[("M", current_dataset, pred_len, model)] = metrics
    return rows


def parse_single_existing():
    text = SINGLE_FULL_PATH.read_text()
    rows = {}
    current_dataset = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if (
            not line
            or "\\midrule" in line
            or "\\bottomrule" in line
            or "\\multicolumn" in line
            or "\\cmidrule(lr)" in line
            or "MSE & MAE" in line
            or "& & MSE" in line
        ):
            continue
        if "\\multirow" in line:
            groups = re.findall(r"\{([^}]*)\}", line)
            if groups:
                current_dataset = groups[-1].replace(r"\_", "_")
        if "\\multirow" not in line and not line.startswith("&"):
            continue
        if "&" not in line:
            continue
        parts = [part.strip() for part in line.rstrip("\\").split("&")]
        if len(parts) < 3:
            continue
        if current_dataset is None:
            continue
        horizon = parts[1].strip()
        key = (current_dataset, "Avg") if "Avg" in horizon else (current_dataset, int(horizon))
        rows[key] = parts[2:6]
    return rows


def parse_single_new_model_metrics():
    if not SINGLE_FULL_PATH.exists():
        return {}
    text = SINGLE_FULL_PATH.read_text()
    rows = {}
    current_dataset = None
    model_offsets = {"Crossformer": 6, "DLinear": 8, "TimesNet": 10}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if (
            not line
            or "\\midrule" in line
            or "\\bottomrule" in line
            or "\\multicolumn" in line
            or "\\cmidrule(lr)" in line
            or "MSE & MAE" in line
            or "& & MSE" in line
        ):
            continue
        if "\\multirow" in line:
            groups = re.findall(r"\{([^}]*)\}", line)
            if groups:
                current_dataset = groups[-1].replace(r"\_", "_")
        if "\\multirow" not in line and not line.startswith("&"):
            continue
        if "&" not in line or current_dataset is None:
            continue
        parts = [part.strip() for part in line.rstrip("\\").split("&")]
        if len(parts) < 12:
            continue
        horizon = parts[1].strip()
        if "Avg" in horizon:
            continue
        pred_len = int(horizon)
        for model, offset in model_offsets.items():
            metrics = metric_pair_from_cells(parts[offset], parts[offset + 1])
            if metrics is not None:
                rows[("MS", current_dataset, pred_len, model)] = metrics
    return rows


def existing_table_metrics():
    metrics = {}
    metrics.update(parse_multivariate_new_model_metrics())
    metrics.update(parse_single_new_model_metrics())
    return metrics


def build_multivariate_table(completed):
    existing = parse_multivariate_existing()
    lines = []
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"\caption{Full multivariate forecasting results. We compare our model (denoted as \textbf{Ours}) with state-of-the-art baselines. For ETT, Exchange, Weather, and futures datasets, $H \in \{96, 192, 336, 720\}$; for ILI, $H \in \{24, 36, 48, 60\}$. The \textbf{\textcolor{red}{best}} results are in \textbf{\textcolor{red}{red bold}}, and the \underline{second best} are underlined. Lower MSE/MAE is better.}")
    lines.append(r"\label{tab:multivariate_full}")
    lines.append(r"\centering")
    lines.append(r"\newcommand{\best}[1]{\textcolor{red}{\textbf{#1}}}")
    lines.append(r"\newcommand{\second}[1]{\underline{#1}}")
    lines.append("")
    lines.append(r"\resizebox{0.9\textwidth}{!}{")
    lines.append(r"\begin{tabular}{c|c|cc|cc|cc|cc|cc|cc|cc|cc|cc}")
    lines.append(r"\toprule")
    lines.append(r"\multicolumn{1}{c|}{\multirow{2}{*}{Dataset}} & \multirow{2}{*}{Len} & \multicolumn{2}{c|}{\textbf{Ours}} & \multicolumn{2}{c|}{TimeXer} & \multicolumn{2}{c|}{TimeMixer} & \multicolumn{2}{c|}{iTransformer} & \multicolumn{2}{c|}{PatchTST} & \multicolumn{2}{c|}{Autoformer} & \multicolumn{2}{c|}{Crossformer} & \multicolumn{2}{c|}{DLinear} & \multicolumn{2}{c}{TimesNet} \\")
    lines.append(r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10} \cmidrule(lr){11-12} \cmidrule(lr){13-14} \cmidrule(lr){15-16} \cmidrule(lr){17-18} \cmidrule(lr){19-20}")
    lines.append(r" & & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE \\")
    lines.append(r"\midrule")
    lines.append("")
    for dataset in MULTI_DATASET_ORDER:
        lines.append(f"% === {dataset} ===")
        horizons = HORIZONS[dataset]
        for idx, pred_len in enumerate(horizons):
            prefix = rf"\multirow{{{len(horizons)}}}{{*}}{{{LATEX_DATASET[dataset]}}} " if idx == 0 else ""
            old_cells = existing[(dataset, pred_len)]
            new_cells = []
            for model in NEW_MODELS:
                metrics = completed.get(("M", dataset, pred_len, model))
                new_cells.extend([metric_or_dash(metrics, "mse"), metric_or_dash(metrics, "mae")])
            line = f"{prefix}& {pred_len} & " + " & ".join(old_cells + new_cells) + r" \\"
            lines.append(line)
        if dataset != MULTI_DATASET_ORDER[-1]:
            lines.append(r"\midrule")
            lines.append("")
    lines.append("")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    lines.append("")
    return "\n".join(lines)


def build_single_table(completed):
    existing = parse_single_existing()
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"  \caption{Full univariate forecasting results for different prediction lengths. For ETT, Exchange, Weather, and futures datasets, $H \in \{96, 192, 336, 720\}$; for ILI, $H \in \{24, 36, 48, 60\}$. We utilize TimeXer as the primary SOTA baseline for exogenous-variable assisted forecasting. \textbf{Bold} highlights the superior results.}")
    lines.append(r"  \label{tab:univariate_full}")
    lines.append(r"  \newcommand{\best}[1]{\textcolor{red}{\textbf{#1}}}")
    lines.append(r"  \centering")
    lines.append(r"  \resizebox{0.45\textwidth}{!}{")
    lines.append(r"  \begin{tabular}{c|c|cc|cc|cc|cc|cc}")
    lines.append(r"    \toprule")
    lines.append(r"    \multirow{2}{*}{Dataset} & \multirow{2}{*}{$H$} & \multicolumn{2}{c|}{\textbf{Ours}} & \multicolumn{2}{c|}{TimeXer} & \multicolumn{2}{c|}{Crossformer} & \multicolumn{2}{c|}{DLinear} & \multicolumn{2}{c}{TimesNet} \\")
    lines.append(r"    \cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10} \cmidrule(lr){11-12}")
    lines.append(r"    & & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE \\")
    lines.append(r"    \midrule")
    for dataset in SINGLE_DATASET_ORDER:
        horizons = HORIZONS[dataset]
        for idx, pred_len in enumerate(horizons):
            prefix = rf"    \multirow{{{len(horizons) + 1}}}{{*}}{{{dataset.replace('_', r'\_')}}}" if idx == 0 else "   "
            old_cells = existing[(dataset, pred_len)]
            new_cells = []
            for model in NEW_MODELS:
                metrics = completed.get(("MS", dataset, pred_len, model))
                new_cells.extend([metric_or_dash(metrics, "mse"), metric_or_dash(metrics, "mae")])
            lines.append(f"{prefix}\n    & {pred_len}  & " + " & ".join(old_cells + new_cells) + r" \\")
        avg_old = existing[(dataset, "Avg")]
        averages = []
        for model in NEW_MODELS:
            metric_list = [completed.get(("MS", dataset, pred_len, model)) for pred_len in horizons]
            averages.append(average_metrics(metric_list))
        new_cells = []
        for metrics in averages:
            new_cells.extend([metric_or_dash(metrics, "mse"), metric_or_dash(metrics, "mae")])
        lines.append(r"    \cmidrule{2-12}")
        lines.append("    & \\textbf{Avg} & " + " & ".join(avg_old + new_cells) + r" \\")
        if dataset != SINGLE_DATASET_ORDER[-1]:
            lines.append(r"    \midrule")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"  }")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def update_tables(jobs):
    completed = completed_metrics_by_job(jobs)
    ALL_RESULTS_PATH.write_text(build_multivariate_table(completed))
    SINGLE_FULL_PATH.write_text(build_single_table(completed))


def launch_jobs(jobs, gpus, poll_secs, update_tex, torch_threads, interop_threads):
    LOG_DIR.mkdir(exist_ok=True)
    completed = completed_metrics_by_job(jobs)
    pending = []
    for job in jobs:
        if job_key(job) not in completed:
            pending.append(job)
    running = {}

    write_state(jobs, pending, running)
    write_report(jobs, pending, running)

    while pending or running:
        while pending and len(running) < len(gpus):
            physical_gpu = next(item for item in gpus if item not in running)
            job = pending.pop(0)
            setting = setting_from_args(job["args"])
            log_path = LOG_DIR / f"{setting}.log"
            command = job_to_command(job["args"], gpu=0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
            env["OMP_NUM_THREADS"] = str(torch_threads)
            env["MKL_NUM_THREADS"] = str(torch_threads)
            env["OPENBLAS_NUM_THREADS"] = str(torch_threads)
            env["NUMEXPR_NUM_THREADS"] = str(torch_threads)
            env["VECLIB_MAXIMUM_THREADS"] = str(torch_threads)
            env["BLIS_NUM_THREADS"] = str(torch_threads)
            env["TSL_TORCH_NUM_THREADS"] = str(torch_threads)
            env["TSL_TORCH_INTEROP_THREADS"] = str(interop_threads)
            env["TSL_MINIMAL_TEST_OUTPUT"] = "1"
            log_handle = log_path.open("w")
            log_handle.write(" ".join(shlex.quote(part) for part in command) + "\n\n")
            log_handle.write(f"CUDA_VISIBLE_DEVICES={physical_gpu}\n\n")
            log_handle.write(
                "THREAD_ENV "
                f"OMP_NUM_THREADS={env['OMP_NUM_THREADS']} "
                f"MKL_NUM_THREADS={env['MKL_NUM_THREADS']} "
                f"OPENBLAS_NUM_THREADS={env['OPENBLAS_NUM_THREADS']} "
                f"TSL_TORCH_NUM_THREADS={env['TSL_TORCH_NUM_THREADS']} "
                f"TSL_TORCH_INTEROP_THREADS={env['TSL_TORCH_INTEROP_THREADS']}\n\n"
            )
            log_handle.flush()
            proc = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=env,
            )
            running[physical_gpu] = {
                "job": job,
                "proc": proc,
                "log_path": log_path,
                "log_handle": log_handle,
                "setting": setting,
                "launched_at": now_utc(),
            }

        write_state(jobs, pending, running)
        write_report(jobs, pending, running)
        if update_tex and completed_metrics_by_job(jobs):
            update_tables(jobs)

        time.sleep(poll_secs)

        finished = []
        for gpu, item in running.items():
            returncode = item["proc"].poll()
            if returncode is None:
                continue
            item["log_handle"].flush()
            item["log_handle"].close()
            metrics = result_metrics(item["setting"])
            status = "completed" if returncode == 0 and metrics is not None else "failed"
            record_completion(item["job"], status, returncode, item["log_path"], metrics=metrics)
            finished.append(gpu)
        for gpu in finished:
            running.pop(gpu, None)

    write_state(jobs, pending, running)
    write_report(jobs, pending, running)
    if update_tex and completed_metrics_by_job(jobs):
        update_tables(jobs)


def dry_run(jobs, limit):
    selected = jobs if limit is None else jobs[:limit]
    for job in selected:
        setting = setting_from_args(job["args"])
        command = job_to_command(job["args"], gpu=0)
        print(json.dumps(
            {
                "mode": job["mode"],
                "dataset": job["dataset"],
                "pred_len": job["pred_len"],
                "model": job["model"],
                "setting": setting,
                "command": " ".join(shlex.quote(part) for part in command),
            },
            ensure_ascii=False,
        ))


def parse_args():
    parser = argparse.ArgumentParser(description="Run Crossformer/DLinear/TimesNet completion experiments.")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6", help="Comma-separated GPU ids.")
    parser.add_argument(
        "--models",
        default="DLinear,Crossformer",
        help="Comma-separated models to schedule from {DLinear,Crossformer,TimesNet}.",
    )
    parser.add_argument("--poll-secs", type=int, default=30, help="Scheduler polling interval.")
    parser.add_argument("--num-workers", type=int, default=1, help="DataLoader workers per run.")
    parser.add_argument("--torch-threads", type=int, default=1, help="Torch/BLAS CPU threads per run.")
    parser.add_argument("--interop-threads", type=int, default=1, help="Torch inter-op CPU threads per run.")
    parser.add_argument("--limit", type=int, default=None, help="Only launch the first N pending jobs.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned jobs and exit.")
    parser.add_argument("--no-update-tex", action="store_true", help="Skip updating tex tables.")
    return parser.parse_args()


def main():
    args = parse_args()
    selected_models = [item.strip() for item in args.models.split(",") if item.strip()]
    unknown = [item for item in selected_models if item not in NEW_MODELS]
    if unknown:
        raise ValueError(f"Unknown models in --models: {unknown}")
    jobs = build_jobs(selected_models=selected_models, num_workers=args.num_workers)
    if args.limit is not None:
        jobs = jobs[:args.limit]
    if args.dry_run:
        dry_run(jobs, args.limit)
        return
    gpus = [int(item.strip()) for item in args.gpus.split(",") if item.strip()]
    launch_jobs(
        jobs,
        gpus,
        args.poll_secs,
        update_tex=not args.no_update_tex,
        torch_threads=args.torch_threads,
        interop_threads=args.interop_threads,
    )


if __name__ == "__main__":
    main()
