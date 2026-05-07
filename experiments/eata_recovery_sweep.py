#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
OLD_RUNS_PATH = ROOT / "study_runs.jsonl"
REPORT_PATH = ROOT / "EATA_recovery_report.md"
RUNS_PATH = ROOT / "recovery_runs.jsonl"
STATE_PATH = ROOT / "recovery_state.json"
LOG_DIR = ROOT / "recovery_logs"


RUN_DEFAULTS = {
    "task_name": "long_term_forecast",
    "is_training": "1",
    "num_workers": "2",
    "des": "RECOVER",
    "dropout": "0.1",
    "batch_size": "32",
    "learning_rate": "0.0001",
    "patience": "5",
    "train_epochs": "20",
    "itr": "1",
    "d_model": "128",
    "k_lookback": "64",
    "hidden": "28",
}


FUTURES_MS_BASELINES = {
    "Futures_TA": {
        96: ("TimeXer", 0.010, 0.068),
        192: ("TimeXer", 0.020, 0.099),
        336: ("TimeXer", 0.035, 0.134),
        720: ("TimeXer", 0.077, 0.209),
    },
    "Futures_RB": {
        96: ("TimeXer", 0.006, 0.056),
        192: ("TimeXer", 0.011, 0.080),
        336: ("TimeXer", 0.020, 0.107),
        720: ("TimeXer", 0.045, 0.159),
    },
}


ILI_M_BASELINES = {
    24: ("TimeMixer", 2.603, 0.960),
    36: ("TimeMixer", 2.261, 0.938),
    48: ("TimeXer", 2.179, 0.901),
    60: ("TimeMixer", 2.147, 0.925),
}


WEATHER_M_REPORTED = {
    96: (0.162, 0.211),
    192: (0.208, 0.254),
    336: (0.262, 0.293),
    720: (0.341, 0.343),
}


FOCUS_GROUPS = [
    ("MS", "Futures_TA", [96, 192, 336, 720]),
    ("MS", "Futures_RB", [96, 192, 336, 720]),
    ("M", "Weather", [96, 192, 336, 720]),
    ("M", "ILI", [24, 36, 48, 60]),
]


FUTURES_MS_VARIANTS = {
    96: [
        {"variant": "short_anchor", "d_model": "128", "dropout": "0.20", "batch_size": "16", "learning_rate": "0.00375", "k_lookback": "64", "hidden": "28"},
        {"variant": "short_wider", "d_model": "160", "dropout": "0.18", "batch_size": "12", "learning_rate": "0.00300", "k_lookback": "64", "hidden": "32"},
        {"variant": "short_longlb", "d_model": "128", "dropout": "0.15", "batch_size": "12", "learning_rate": "0.00450", "k_lookback": "80", "hidden": "32"},
        {"variant": "short_deep", "d_model": "192", "dropout": "0.20", "batch_size": "8", "learning_rate": "0.00250", "k_lookback": "64", "hidden": "36"},
        {"variant": "short_reg", "d_model": "128", "dropout": "0.25", "batch_size": "24", "learning_rate": "0.00300", "k_lookback": "48", "hidden": "28"},
        {"variant": "short_seq", "d_model": "96", "dropout": "0.15", "batch_size": "16", "learning_rate": "0.00500", "k_lookback": "96", "hidden": "24"},
    ],
    192: [
        {"variant": "mid_anchor", "d_model": "128", "dropout": "0.05", "batch_size": "16", "learning_rate": "0.000563", "k_lookback": "64", "hidden": "28"},
        {"variant": "mid_wider", "d_model": "160", "dropout": "0.05", "batch_size": "12", "learning_rate": "0.000650", "k_lookback": "64", "hidden": "32"},
        {"variant": "mid_lowerlr", "d_model": "128", "dropout": "0.08", "batch_size": "12", "learning_rate": "0.000450", "k_lookback": "80", "hidden": "32"},
        {"variant": "mid_deep", "d_model": "192", "dropout": "0.05", "batch_size": "8", "learning_rate": "0.000700", "k_lookback": "64", "hidden": "36"},
        {"variant": "mid_seq", "d_model": "128", "dropout": "0.03", "batch_size": "16", "learning_rate": "0.000800", "k_lookback": "96", "hidden": "28"},
        {"variant": "mid_compact", "d_model": "96", "dropout": "0.10", "batch_size": "24", "learning_rate": "0.000500", "k_lookback": "64", "hidden": "24"},
    ],
    336: [
        {"variant": "long_anchor", "d_model": "128", "dropout": "0.05", "batch_size": "16", "learning_rate": "0.000750", "k_lookback": "64", "hidden": "28"},
        {"variant": "long_wider", "d_model": "160", "dropout": "0.05", "batch_size": "12", "learning_rate": "0.000650", "k_lookback": "64", "hidden": "32"},
        {"variant": "long_seq", "d_model": "128", "dropout": "0.03", "batch_size": "16", "learning_rate": "0.000850", "k_lookback": "96", "hidden": "28"},
        {"variant": "long_deep", "d_model": "192", "dropout": "0.05", "batch_size": "8", "learning_rate": "0.000600", "k_lookback": "80", "hidden": "36"},
        {"variant": "long_reg", "d_model": "128", "dropout": "0.08", "batch_size": "12", "learning_rate": "0.000500", "k_lookback": "64", "hidden": "32"},
        {"variant": "long_compact", "d_model": "96", "dropout": "0.10", "batch_size": "24", "learning_rate": "0.000550", "k_lookback": "96", "hidden": "24"},
    ],
    720: [
        {"variant": "xl_anchor", "d_model": "128", "dropout": "0.05", "batch_size": "16", "learning_rate": "0.000750", "k_lookback": "64", "hidden": "28"},
        {"variant": "xl_wider", "d_model": "160", "dropout": "0.05", "batch_size": "12", "learning_rate": "0.000650", "k_lookback": "64", "hidden": "32"},
        {"variant": "xl_seq", "d_model": "128", "dropout": "0.03", "batch_size": "16", "learning_rate": "0.000850", "k_lookback": "96", "hidden": "28"},
        {"variant": "xl_deep", "d_model": "192", "dropout": "0.05", "batch_size": "8", "learning_rate": "0.000550", "k_lookback": "80", "hidden": "36"},
        {"variant": "xl_reg", "d_model": "128", "dropout": "0.08", "batch_size": "12", "learning_rate": "0.000450", "k_lookback": "64", "hidden": "32"},
        {"variant": "xl_compact", "d_model": "96", "dropout": "0.10", "batch_size": "24", "learning_rate": "0.000500", "k_lookback": "96", "hidden": "24"},
    ],
}


ILI_M_VARIANTS = {
    24: [
        {"variant": "ili24_fast", "d_model": "128", "dropout": "0.05", "batch_size": "16", "learning_rate": "0.0375", "k_lookback": "36", "hidden": "28"},
        {"variant": "ili24_balanced", "d_model": "96", "dropout": "0.08", "batch_size": "24", "learning_rate": "0.0200", "k_lookback": "36", "hidden": "32"},
        {"variant": "ili24_shortlb", "d_model": "64", "dropout": "0.10", "batch_size": "32", "learning_rate": "0.0250", "k_lookback": "24", "hidden": "28"},
        {"variant": "ili24_deep", "d_model": "192", "dropout": "0.05", "batch_size": "8", "learning_rate": "0.0200", "k_lookback": "36", "hidden": "40"},
        {"variant": "ili24_midreg", "d_model": "128", "dropout": "0.12", "batch_size": "12", "learning_rate": "0.0150", "k_lookback": "30", "hidden": "36"},
        {"variant": "ili24_small", "d_model": "64", "dropout": "0.06", "batch_size": "24", "learning_rate": "0.0300", "k_lookback": "30", "hidden": "24"},
    ],
    36: [
        {"variant": "ili36_fast", "d_model": "128", "dropout": "0.10", "batch_size": "12", "learning_rate": "0.0150", "k_lookback": "36", "hidden": "32"},
        {"variant": "ili36_balanced", "d_model": "96", "dropout": "0.15", "batch_size": "16", "learning_rate": "0.0120", "k_lookback": "24", "hidden": "32"},
        {"variant": "ili36_small", "d_model": "64", "dropout": "0.20", "batch_size": "24", "learning_rate": "0.0080", "k_lookback": "36", "hidden": "28"},
        {"variant": "ili36_deep", "d_model": "192", "dropout": "0.10", "batch_size": "8", "learning_rate": "0.0100", "k_lookback": "24", "hidden": "40"},
        {"variant": "ili36_reg", "d_model": "128", "dropout": "0.18", "batch_size": "8", "learning_rate": "0.0060", "k_lookback": "36", "hidden": "36"},
        {"variant": "ili36_soft", "d_model": "64", "dropout": "0.12", "batch_size": "32", "learning_rate": "0.0065", "k_lookback": "30", "hidden": "24"},
    ],
    48: [
        {"variant": "ili48_anchor", "d_model": "128", "dropout": "0.20", "batch_size": "8", "learning_rate": "0.0050", "k_lookback": "36", "hidden": "32"},
        {"variant": "ili48_wider", "d_model": "192", "dropout": "0.25", "batch_size": "6", "learning_rate": "0.0030", "k_lookback": "36", "hidden": "40"},
        {"variant": "ili48_shortlb", "d_model": "96", "dropout": "0.25", "batch_size": "12", "learning_rate": "0.0040", "k_lookback": "24", "hidden": "28"},
        {"variant": "ili48_sharp", "d_model": "128", "dropout": "0.15", "batch_size": "8", "learning_rate": "0.0060", "k_lookback": "36", "hidden": "36"},
        {"variant": "ili48_small", "d_model": "64", "dropout": "0.22", "batch_size": "16", "learning_rate": "0.0045", "k_lookback": "36", "hidden": "28"},
        {"variant": "ili48_mid", "d_model": "160", "dropout": "0.18", "batch_size": "6", "learning_rate": "0.0035", "k_lookback": "30", "hidden": "32"},
    ],
    60: [
        {"variant": "ili60_anchor", "d_model": "128", "dropout": "0.10", "batch_size": "8", "learning_rate": "0.0010", "k_lookback": "36", "hidden": "32"},
        {"variant": "ili60_wider", "d_model": "192", "dropout": "0.15", "batch_size": "6", "learning_rate": "0.00075", "k_lookback": "36", "hidden": "40"},
        {"variant": "ili60_shortlb", "d_model": "96", "dropout": "0.15", "batch_size": "12", "learning_rate": "0.0015", "k_lookback": "24", "hidden": "28"},
        {"variant": "ili60_lowlr", "d_model": "128", "dropout": "0.20", "batch_size": "8", "learning_rate": "0.00050", "k_lookback": "36", "hidden": "36"},
        {"variant": "ili60_small", "d_model": "64", "dropout": "0.10", "batch_size": "16", "learning_rate": "0.0020", "k_lookback": "36", "hidden": "28"},
        {"variant": "ili60_mid", "d_model": "160", "dropout": "0.12", "batch_size": "6", "learning_rate": "0.00090", "k_lookback": "30", "hidden": "32"},
    ],
}


WEATHER_M_VARIANTS = {
    96: [
        {"variant": "wx96_anchor", "d_model": "512", "dropout": "0.05", "batch_size": "32", "learning_rate": "0.00010", "k_lookback": "8", "hidden": "10"},
        {"variant": "wx96_largebs", "d_model": "256", "dropout": "0.05", "batch_size": "64", "learning_rate": "0.00015", "k_lookback": "8", "hidden": "16"},
        {"variant": "wx96_shortlb", "d_model": "512", "dropout": "0.08", "batch_size": "64", "learning_rate": "0.00015", "k_lookback": "4", "hidden": "16"},
        {"variant": "wx96_mid", "d_model": "256", "dropout": "0.10", "batch_size": "128", "learning_rate": "0.00020", "k_lookback": "16", "hidden": "24"},
        {"variant": "wx96_compact", "d_model": "128", "dropout": "0.10", "batch_size": "128", "learning_rate": "0.00050", "k_lookback": "32", "hidden": "24"},
    ],
    192: [
        {"variant": "wx192_anchor", "d_model": "512", "dropout": "0.05", "batch_size": "32", "learning_rate": "0.00010", "k_lookback": "4", "hidden": "10"},
        {"variant": "wx192_largebs", "d_model": "256", "dropout": "0.05", "batch_size": "64", "learning_rate": "0.00012", "k_lookback": "8", "hidden": "16"},
        {"variant": "wx192_shortlb", "d_model": "512", "dropout": "0.08", "batch_size": "64", "learning_rate": "0.00010", "k_lookback": "4", "hidden": "16"},
        {"variant": "wx192_mid", "d_model": "256", "dropout": "0.10", "batch_size": "128", "learning_rate": "0.00018", "k_lookback": "16", "hidden": "24"},
        {"variant": "wx192_compact", "d_model": "128", "dropout": "0.10", "batch_size": "128", "learning_rate": "0.00035", "k_lookback": "32", "hidden": "24"},
    ],
    336: [
        {"variant": "wx336_anchor", "d_model": "512", "dropout": "0.05", "batch_size": "32", "learning_rate": "0.00010", "k_lookback": "4", "hidden": "10"},
        {"variant": "wx336_largebs", "d_model": "256", "dropout": "0.05", "batch_size": "64", "learning_rate": "0.00012", "k_lookback": "8", "hidden": "16"},
        {"variant": "wx336_shortlb", "d_model": "512", "dropout": "0.08", "batch_size": "64", "learning_rate": "0.00010", "k_lookback": "4", "hidden": "16"},
        {"variant": "wx336_mid", "d_model": "256", "dropout": "0.10", "batch_size": "128", "learning_rate": "0.00015", "k_lookback": "16", "hidden": "24"},
        {"variant": "wx336_compact", "d_model": "128", "dropout": "0.10", "batch_size": "128", "learning_rate": "0.00030", "k_lookback": "32", "hidden": "24"},
    ],
    720: [
        {"variant": "wx720_anchor", "d_model": "512", "dropout": "0.05", "batch_size": "32", "learning_rate": "0.00030", "k_lookback": "4", "hidden": "10"},
        {"variant": "wx720_largebs", "d_model": "256", "dropout": "0.05", "batch_size": "64", "learning_rate": "0.00020", "k_lookback": "8", "hidden": "16"},
        {"variant": "wx720_shortlb", "d_model": "512", "dropout": "0.08", "batch_size": "64", "learning_rate": "0.00025", "k_lookback": "4", "hidden": "16"},
        {"variant": "wx720_mid", "d_model": "256", "dropout": "0.10", "batch_size": "128", "learning_rate": "0.00020", "k_lookback": "16", "hidden": "24"},
        {"variant": "wx720_compact", "d_model": "128", "dropout": "0.10", "batch_size": "128", "learning_rate": "0.00040", "k_lookback": "32", "hidden": "24"},
    ],
}


GROUP_PRIORITY = {
    ("MS", "Futures_TA"): 0,
    ("MS", "Futures_RB"): 1,
    ("M", "Weather"): 2,
    ("M", "ILI"): 3,
}


def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


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
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "mape": float(mape),
        "mspe": float(mspe),
    }


def normalize_model_name(name):
    return "EATA" if name in {"EATA", "TD_CaA"} else name


def make_job(args, dataset, task_type, pred_len, variant, source):
    merged = deepcopy(RUN_DEFAULTS)
    merged.update(args)
    merged["pred_len"] = str(pred_len)
    merged["features"] = task_type
    return {
        "dataset": dataset,
        "task_type": task_type,
        "pred_len": int(pred_len),
        "variant": variant,
        "source": source,
        "args": merged,
        "model": normalize_model_name(merged["model"]),
    }


def futures_ms_base(dataset, pred_len):
    suffix = dataset.split("_", 1)[1]
    return {
        "model": "EATA",
        "task_name": "long_term_forecast",
        "is_training": "1",
        "root_path": "./dataset/futures/",
        "data_path": f"futures_{suffix}.csv",
        "model_id": f"{dataset}_96_{pred_len}",
        "data": "Futures",
        "target": "LastPrice",
        "features": "MS",
        "seq_len": "96",
        "label_len": "48",
        "pred_len": str(pred_len),
        "e_layers": "1",
        "factor": "3",
        "enc_in": "146",
        "dec_in": "146",
        "c_out": "1",
        "train_epochs": "30",
        "patience": "5",
        "method": "Dynamic",
        "bias": True,
        "interact": True,
        "des": "REC-MS",
    }


def ili_m_base(pred_len):
    return {
        "model": "EATA",
        "task_name": "long_term_forecast",
        "is_training": "1",
        "root_path": "./dataset/illness/",
        "data_path": "national_illness.csv",
        "model_id": f"ili_36_{pred_len}",
        "data": "custom",
        "target": "OT",
        "features": "M",
        "seq_len": "36",
        "label_len": "18",
        "pred_len": str(pred_len),
        "e_layers": "4",
        "d_layers": "1",
        "factor": "3",
        "enc_in": "7",
        "dec_in": "7",
        "c_out": "7",
        "moving_avg": "10",
        "train_epochs": "20",
        "patience": "5",
        "method": "Dynamic",
        "bias": True,
        "interact": True,
        "des": "REC-M",
    }


def weather_m_base(pred_len):
    return {
        "model": "EATA",
        "task_name": "long_term_forecast",
        "is_training": "1",
        "root_path": "./dataset/weather/",
        "data_path": "weather.csv",
        "model_id": f"weather_96_{pred_len}",
        "data": "custom",
        "features": "M",
        "seq_len": "96",
        "label_len": "48",
        "pred_len": str(pred_len),
        "e_layers": "1",
        "factor": "3",
        "enc_in": "21",
        "dec_in": "21",
        "c_out": "21",
        "d_ff": "512",
        "moving_avg": "1",
        "train_epochs": "25",
        "patience": "5",
        "method": "Dynamic",
        "bias": True,
        "interact": True,
        "des": "REC-M",
    }


def build_queue():
    jobs = []
    for dataset in ["Futures_TA", "Futures_RB"]:
        for pred_len, variants in FUTURES_MS_VARIANTS.items():
            base = futures_ms_base(dataset, pred_len)
            for variant in variants:
                args = deepcopy(base)
                args.update({k: v for k, v in variant.items() if k != "variant"})
                jobs.append(make_job(args, dataset, "MS", pred_len, variant["variant"], "focused:futures_ms"))

    for pred_len, variants in ILI_M_VARIANTS.items():
        base = ili_m_base(pred_len)
        for variant in variants:
            args = deepcopy(base)
            args.update({k: v for k, v in variant.items() if k != "variant"})
            jobs.append(make_job(args, "ILI", "M", pred_len, variant["variant"], "focused:ili_m"))

    for pred_len, variants in WEATHER_M_VARIANTS.items():
        base = weather_m_base(pred_len)
        for variant in variants:
            args = deepcopy(base)
            args.update({k: v for k, v in variant.items() if k != "variant"})
            jobs.append(make_job(args, "Weather", "M", pred_len, variant["variant"], "focused:weather_m"))

    deduped = {}
    for job in jobs:
        deduped[setting_from_args(job["args"])] = job
    return sorted(deduped.values(), key=job_sort_key)


def job_sort_key(job):
    return (
        GROUP_PRIORITY.get((job["task_type"], job["dataset"]), 99),
        job["pred_len"],
        job["variant"],
        setting_from_args(job["args"]),
    )


def load_jsonl_records(path):
    records = []
    if not path.exists():
        return records
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def is_focus_record(record):
    if normalize_model_name(record.get("model", "")) != "EATA":
        return False
    task_type = record.get("task_type")
    dataset = record.get("dataset")
    pred_len = int(record.get("pred_len", -1))
    for focus_task, focus_dataset, horizons in FOCUS_GROUPS:
        if task_type == focus_task and dataset == focus_dataset and pred_len in horizons:
            return True
    return False


def load_previous_records():
    records = []
    for record in load_jsonl_records(OLD_RUNS_PATH):
        if record.get("status") == "completed" and record.get("metrics") and is_focus_record(record):
            record["model"] = normalize_model_name(record["model"])
            records.append(record)
    return records


def load_recovery_records():
    return [r for r in load_jsonl_records(RUNS_PATH) if r.get("status") == "completed" and r.get("metrics")]


def best_by_key(records):
    best = {}
    for record in records:
        metrics = record.get("metrics")
        if not metrics:
            continue
        key = (record["task_type"], record["dataset"], int(record["pred_len"]))
        cur = best.get(key)
        if cur is None:
            best[key] = record
            continue
        cur_metrics = cur["metrics"]
        if metrics["mse"] < cur_metrics["mse"] or (
            abs(metrics["mse"] - cur_metrics["mse"]) < 1e-12 and metrics["mae"] < cur_metrics["mae"]
        ):
            best[key] = record
    return best


def format_pair(pair):
    if pair is None:
        return "-", "-"
    return f"{pair[0]:.3f}", f"{pair[1]:.3f}"


def format_record_metrics(record):
    if not record:
        return "-", "-"
    return f"{record['metrics']['mse']:.3f}", f"{record['metrics']['mae']:.3f}"


def param_summary(record):
    if not record:
        return "-"
    args = record["args"]
    return (
        f"d_model={args.get('d_model')} "
        f"dropout={args.get('dropout')} "
        f"k_lookback={args.get('k_lookback')} "
        f"lr={args.get('learning_rate')} "
        f"bs={args.get('batch_size')}"
    )


def compare_against_baseline(record, baseline_mse, baseline_mae):
    if not record:
        return "running"
    mse = record["metrics"]["mse"]
    mae = record["metrics"]["mae"]
    if mse < baseline_mse and mae <= baseline_mae:
        return "beat"
    if mse <= baseline_mse and mae < baseline_mae:
        return "beat"
    if mse <= baseline_mse * 1.03 and mae <= baseline_mae * 1.03:
        return "close"
    return "behind"


def append_run_record(record):
    with RUNS_PATH.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def append_completion_record(job, status, metrics, returncode, log_path="", notes=""):
    record = {
        "status": status,
        "finished_at": now_utc(),
        "setting": setting_from_args(job["args"]),
        "dataset": job["dataset"],
        "task_type": job["task_type"],
        "pred_len": job["pred_len"],
        "model": job["model"],
        "variant": job["variant"],
        "source": job["source"],
        "notes": notes,
        "args": job["args"],
        "metrics": metrics,
        "returncode": returncode,
        "log_path": log_path,
    }
    append_run_record(record)
    write_report()


def record_completed_job(job, returncode, log_path):
    setting = setting_from_args(job["args"])
    metrics = result_metrics(setting)
    append_completion_record(
        job,
        "completed" if returncode == 0 and metrics is not None else "failed",
        metrics,
        returncode,
        str(log_path.relative_to(ROOT)),
    )


def existing_completed_settings():
    completed = set()
    for record in load_jsonl_records(RUNS_PATH):
        if record.get("status") == "completed":
            completed.add(record["setting"])
    return completed


def running_payload(running):
    return [
        {
            "gpu": gpu,
            "dataset": item["job"]["dataset"],
            "task_type": item["job"]["task_type"],
            "pred_len": item["job"]["pred_len"],
            "variant": item["job"]["variant"],
            "setting": item["setting"],
        }
        for gpu, item in sorted(running.items())
    ]


def write_state(running_jobs, pending_count):
    STATE_PATH.write_text(
        json.dumps(
            {
                "updated_at": now_utc(),
                "pending_jobs": pending_count,
                "running_jobs": running_jobs,
            },
            indent=2,
        )
    )


def write_report():
    queue = build_queue()
    state = {"updated_at": now_utc(), "pending_jobs": 0, "running_jobs": []}
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text())
    else:
        completed = existing_completed_settings()
        pending = 0
        for job in queue:
            setting = setting_from_args(job["args"])
            if setting in completed or result_metrics(setting) is not None:
                continue
            pending += 1
        state["pending_jobs"] = pending

    previous_records = load_previous_records()
    recovery_records = load_recovery_records()
    all_records = previous_records + recovery_records
    previous_best = best_by_key(previous_records)
    current_best = best_by_key(all_records)

    lines = []
    lines.append("# EATA Recovery Report")
    lines.append("")
    lines.append(f"Updated: {state.get('updated_at', now_utc())}")
    lines.append(f"Pending jobs: {state.get('pending_jobs', 0)}")
    lines.append(f"Running jobs: {len(state.get('running_jobs', []))}")
    lines.append("")
    lines.append("This report focuses on the near-gap recovery sweep for `Futures_TA`, `Futures_RB`, `ILI`, and `Weather`.")
    lines.append("")

    lines.append("## Futures MS Recovery")
    lines.append("")
    lines.append("| Dataset | Horizon | TimeXer MSE | TimeXer MAE | Previous EATA MSE | Previous EATA MAE | Current Best MSE | Current Best MAE | Status | Params |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for dataset in ["Futures_TA", "Futures_RB"]:
        for horizon in [96, 192, 336, 720]:
            _, base_mse, base_mae = FUTURES_MS_BASELINES[dataset][horizon]
            prev = previous_best.get(("MS", dataset, horizon))
            cur = current_best.get(("MS", dataset, horizon))
            prev_mse, prev_mae = format_record_metrics(prev)
            cur_mse, cur_mae = format_record_metrics(cur)
            lines.append(
                f"| {dataset} | {horizon} | {base_mse:.3f} | {base_mae:.3f} | {prev_mse} | {prev_mae} | "
                f"{cur_mse} | {cur_mae} | {compare_against_baseline(cur, base_mse, base_mae)} | {param_summary(cur)} |"
            )
    lines.append("")

    lines.append("## ILI M Recovery")
    lines.append("")
    lines.append("| Horizon | Baseline | Baseline MSE | Baseline MAE | Previous EATA MSE | Previous EATA MAE | Current Best MSE | Current Best MAE | Status | Params |")
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for horizon in [24, 36, 48, 60]:
        base_model, base_mse, base_mae = ILI_M_BASELINES[horizon]
        prev = previous_best.get(("M", "ILI", horizon))
        cur = current_best.get(("M", "ILI", horizon))
        prev_mse, prev_mae = format_record_metrics(prev)
        cur_mse, cur_mae = format_record_metrics(cur)
        lines.append(
            f"| {horizon} | {base_model} | {base_mse:.3f} | {base_mae:.3f} | {prev_mse} | {prev_mae} | "
            f"{cur_mse} | {cur_mae} | {compare_against_baseline(cur, base_mse, base_mae)} | {param_summary(cur)} |"
        )
    lines.append("")

    lines.append("## Weather M Recovery")
    lines.append("")
    lines.append("| Horizon | Reported MSE | Reported MAE | Previous EATA MSE | Previous EATA MAE | Current Best MSE | Current Best MAE | Status | Params |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for horizon in [96, 192, 336, 720]:
        base_mse, base_mae = WEATHER_M_REPORTED[horizon]
        prev = previous_best.get(("M", "Weather", horizon))
        cur = current_best.get(("M", "Weather", horizon))
        prev_mse, prev_mae = format_record_metrics(prev)
        cur_mse, cur_mae = format_record_metrics(cur)
        lines.append(
            f"| {horizon} | {base_mse:.3f} | {base_mae:.3f} | {prev_mse} | {prev_mae} | "
            f"{cur_mse} | {cur_mae} | {compare_against_baseline(cur, base_mse, base_mae)} | {param_summary(cur)} |"
        )
    lines.append("")

    lines.append("## Recent Recovery Runs")
    lines.append("")
    lines.append("| Time | Task | Dataset | Horizon | Variant | MSE | MAE | Setting |")
    lines.append("| --- | --- | --- | ---: | --- | ---: | ---: | --- |")
    recent = [r for r in load_jsonl_records(RUNS_PATH) if r.get("status") == "completed"][-30:]
    for record in reversed(recent):
        lines.append(
            f"| {record.get('finished_at', '-')} | {record['task_type']} | {record['dataset']} | {record['pred_len']} | "
            f"{record.get('variant', '-')} | {record['metrics']['mse']:.3f} | {record['metrics']['mae']:.3f} | "
            f"`{record['setting']}` |"
        )
    lines.append("")

    if state.get("running_jobs"):
        lines.append("## Running Jobs")
        lines.append("")
        lines.append("| GPU | Task | Dataset | Horizon | Variant | Setting |")
        lines.append("| --- | --- | --- | ---: | --- | --- |")
        for item in state["running_jobs"]:
            lines.append(
                f"| {item['gpu']} | {item['task_type']} | {item['dataset']} | {item['pred_len']} | "
                f"{item['variant']} | `{item['setting']}` |"
            )
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines))


def job_to_command(args):
    ordered = []
    for key, value in args.items():
        if value is True:
            ordered.append(f"--{key}")
        else:
            ordered.extend([f"--{key}", str(value)])
    return ["python", "run.py"] + ordered


def job_process_alive(job):
    args = job["args"]
    needles = [
        f"--model_id {args.get('model_id', '')}",
        f"--model {args.get('model', '')}",
        f"--des {args.get('des', '')}",
    ]
    try:
        output = subprocess.check_output(["ps", "-eo", "args"], text=True)
    except Exception:
        return False
    for line in output.splitlines():
        if "python" not in line or "run.py" not in line:
            continue
        if all(needle in line for needle in needles):
            return True
    return False


def restore_running_jobs(queue, completed_settings):
    if not STATE_PATH.exists():
        return {}
    try:
        state = json.loads(STATE_PATH.read_text())
    except json.JSONDecodeError:
        return {}

    queue_by_setting = {setting_from_args(job["args"]): job for job in queue}
    restored = {}
    for item in state.get("running_jobs", []):
        setting = item.get("setting")
        gpu = item.get("gpu")
        if setting in completed_settings or gpu is None:
            continue
        metrics = result_metrics(setting)
        if metrics is not None:
            job = queue_by_setting.get(setting)
            if job is not None and setting not in completed_settings:
                append_completion_record(job, "completed", metrics, 0, "", "backfill:restored_running")
                completed_settings.add(setting)
            continue
        job = queue_by_setting.get(setting)
        if job is None or not job_process_alive(job):
            continue
        restored[gpu] = {
            "proc": None,
            "log_file": None,
            "log_path": LOG_DIR / f"{setting}.log",
            "job": job,
            "setting": setting,
            "external": True,
        }
    return restored


def backfill_existing_results(queue, completed_settings):
    for job in queue:
        setting = setting_from_args(job["args"])
        if setting in completed_settings:
            continue
        metrics = result_metrics(setting)
        if metrics is None:
            continue
        append_completion_record(job, "completed", metrics, 0, "", "backfill:existing_result")
        completed_settings.add(setting)


def launch_jobs(gpu_ids, limit=None, poll_secs=20):
    LOG_DIR.mkdir(exist_ok=True)
    queue = build_queue()
    completed_settings = existing_completed_settings()
    backfill_existing_results(queue, completed_settings)
    running = restore_running_jobs(queue, completed_settings)
    reserved_settings = {item["setting"] for item in running.values()}

    pending = []
    for job in queue:
        setting = setting_from_args(job["args"])
        if setting in completed_settings or setting in reserved_settings:
            continue
        pending.append(job)

    if limit is not None:
        pending = pending[:limit]

    write_state(running_payload(running), len(pending))
    write_report()

    while pending or running:
        launched = False
        while pending and len(running) < len(gpu_ids):
            free = [gpu for gpu in gpu_ids if gpu not in running]
            if not free:
                break
            gpu = free[0]
            job = pending.pop(0)
            setting = setting_from_args(job["args"])
            log_path = LOG_DIR / f"{setting}.log"
            cmd = job_to_command(job["args"])
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")
            env.setdefault("OPENBLAS_NUM_THREADS", "1")
            env.setdefault("NUMEXPR_MAX_THREADS", "64")
            log_file = log_path.open("w")
            log_file.write(" ".join(shlex.quote(part) for part in cmd) + "\n\n")
            log_file.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            running[gpu] = {
                "proc": proc,
                "log_file": log_file,
                "log_path": log_path,
                "job": job,
                "setting": setting,
                "external": False,
            }
            launched = True
            write_state(running_payload(running), len(pending))
            write_report()

        finished = []
        for gpu, item in running.items():
            if item.get("external"):
                metrics = result_metrics(item["setting"])
                if metrics is not None:
                    append_completion_record(item["job"], "completed", metrics, 0, "", "backfill:restored_running")
                    finished.append(gpu)
                continue
            returncode = item["proc"].poll()
            if returncode is not None:
                item["log_file"].close()
                record_completed_job(item["job"], returncode, item["log_path"])
                finished.append(gpu)

        for gpu in finished:
            running.pop(gpu, None)

        write_state(running_payload(running), len(pending))
        if running:
            time.sleep(poll_secs)
        elif pending and not launched:
            time.sleep(min(poll_secs, 5))

    write_state([], 0)
    write_report()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["list", "launch", "daemon"], default="launch")
    parser.add_argument("--gpus", default="0,1,2,3,4")
    parser.add_argument("--poll-secs", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    queue = build_queue()
    completed = existing_completed_settings()
    backfill_existing_results(queue, completed)
    write_report()

    if args.mode == "list":
        print(f"queue={len(queue)}")
        for job in queue:
            print(job["task_type"], job["dataset"], job["pred_len"], job["variant"], setting_from_args(job["args"]))
        return

    if args.mode == "daemon":
        LOG_DIR.mkdir(exist_ok=True)
        launcher_log = ROOT / "recovery_launcher.log"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--mode",
            "launch",
            "--gpus",
            args.gpus,
            "--poll-secs",
            str(args.poll_secs),
        ]
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        log_file = launcher_log.open("w")
        log_file.write(" ".join(shlex.quote(part) for part in cmd) + "\n\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=os.environ.copy(),
        )
        print(proc.pid)
        log_file.close()
        return

    gpu_ids = [int(part) for part in args.gpus.split(",") if part.strip()]
    launch_jobs(gpu_ids, limit=args.limit, poll_secs=args.poll_secs)


if __name__ == "__main__":
    main()
