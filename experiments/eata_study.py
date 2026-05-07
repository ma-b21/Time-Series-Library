#!/usr/bin/env python3
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN_PY = ROOT / "run.py"
RESULTS_DIR = ROOT / "results"
REPORT_PATH = ROOT / "EATA_tuning_report.md"
RUNS_PATH = ROOT / "study_runs.jsonl"
STATE_PATH = ROOT / "study_state.json"
LOG_DIR = ROOT / "study_logs"


RUN_DEFAULTS = {
    "task_name": "long_term_forecast",
    "is_training": "1",
    "num_workers": "2",
    "target": "OT",
    "des": "Exp",
    "dropout": "0.1",
    "batch_size": "32",
    "learning_rate": "0.0001",
    "patience": "3",
    "train_epochs": "10",
    "itr": "1",
    "d_model": "512",
    "k_lookback": "64",
    "hidden": "10",
    "label_len": "48",
}


M_TARGETS = {
    "ETTh1": {96: (0.383, 0.391), 192: (0.426, 0.436), 336: (0.458, 0.450), 720: (0.470, 0.466)},
    "ETTh2": {96: (0.287, 0.336), 192: (0.374, 0.391), 336: (0.410, 0.427), 720: (0.414, 0.434)},
    "ETTm1": {96: (0.320, 0.359), 192: (0.362, 0.385), 336: (0.391, 0.405), 720: (0.451, 0.442)},
    "ETTm2": {96: (0.176, 0.260), 192: (0.243, 0.303), 336: (0.305, 0.341), 720: (0.408, 0.399)},
    "Exchange": {96: (0.085, 0.201), 192: (0.178, 0.299), 336: (0.330, 0.415), 720: (0.812, 0.678)},
    "Weather": {96: (0.162, 0.211), 192: (0.208, 0.254), 336: (0.262, 0.293), 720: (0.341, 0.343)},
    "Futures_RB": {96: (0.131, 0.090), 192: (0.146, 0.117), 336: (0.131, 0.135), 720: (0.147, 0.174)},
    "Futures_TA": {96: (0.119, 0.091), 192: (0.182, 0.126), 336: (0.140, 0.134), 720: (0.182, 0.183)},
    "Futures_M": {96: (0.361, 0.212), 192: (0.389, 0.249), 336: (0.411, 0.284), 720: (0.492, 0.363)},
}


MS_TARGETS = {
    "ETTh1": {96: (0.056, 0.178), 192: (0.073, 0.208), 336: (0.082, 0.226), 720: (0.082, 0.225)},
    "ETTh2": {96: (0.128, 0.272), 192: (0.178, 0.327), 336: (0.214, 0.370), 720: (0.215, 0.372)},
    "Weather": {96: (0.001, 0.026), 192: (0.002, 0.030), 336: (0.002, 0.031), 720: (0.002, 0.035)},
    "Exchange": {96: (0.116, 0.251), 192: (0.205, 0.337), 336: (0.420, 0.482), 720: (1.129, 0.820)},
    "Futures_M": {96: (0.006, 0.057), 192: (0.011, 0.080), 336: (0.022, 0.110), 720: (0.054, 0.170)},
}


REQUESTED_M = {
    "ETTh1": [96, 192, 336, 720],
    "ETTh2": [96, 192, 336, 720],
    "ETTm1": [96, 192, 336, 720],
    "ETTm2": [96, 192, 336, 720],
    "Exchange": [96, 192, 336, 720],
    "ILI": [24, 36, 48, 60],
    "Weather": [96, 192, 336, 720],
    "Futures_TA": [96, 192, 336, 720],
    "Futures_RB": [96, 192, 336, 720],
    "Futures_M": [96, 192, 336, 720],
}


REQUESTED_MS = {
    "ETTh1": [96, 192, 336, 720],
    "ETTh2": [96, 192, 336, 720],
    "ETTm1": [96, 192, 336, 720],
    "ETTm2": [96, 192, 336, 720],
    "Exchange": [96, 192, 336, 720],
    "ILI": [24, 36, 48, 60],
    "Weather": [96, 192, 336, 720],
    "Futures_TA": [96, 192, 336, 720],
    "Futures_RB": [96, 192, 336, 720],
    "Futures_M": [96, 192, 336, 720],
}


EXPLICIT_EATA_M_SCRIPTS = [
    ROOT / "scripts/long_term_forecast/ETT_script/EATA_ETTh1.sh",
    ROOT / "scripts/long_term_forecast/ETT_script/EATA_ETTh2.sh",
    ROOT / "scripts/long_term_forecast/ETT_script/EATA_ETTm1.sh",
    ROOT / "scripts/long_term_forecast/ETT_script/EATA_ETTm2.sh",
    ROOT / "scripts/long_term_forecast/Exchange_script/EATA.sh",
    ROOT / "scripts/long_term_forecast/Weather_script/EATA.sh",
    ROOT / "scripts/long_term_forecast/ILI_script/EATA.sh",
    ROOT / "scripts/long_term_forecast/Futures_script/EATA_TA.sh",
]


EXPLICIT_EATA_MS_SCRIPTS = [
    ROOT / "scripts/exogenous_forecast/ETTh1/EATA.sh",
    ROOT / "scripts/exogenous_forecast/ETTh2/EATA.sh",
    ROOT / "scripts/exogenous_forecast/ETTm1/EATA.sh",
    ROOT / "scripts/exogenous_forecast/ETTm2/EATA.sh",
    ROOT / "scripts/exogenous_forecast/Exchange/EATA.sh",
    ROOT / "scripts/exogenous_forecast/Weather/EATA.sh",
]

EXPLICIT_TIMEXER_MS_SCRIPTS = [
    ROOT / "scripts/exogenous_forecast/ETTh1/TimeXer.sh",
    ROOT / "scripts/exogenous_forecast/ETTh2/TimeXer.sh",
    ROOT / "scripts/exogenous_forecast/ETTm1/TimeXer.sh",
    ROOT / "scripts/exogenous_forecast/ETTm2/TimeXer.sh",
    ROOT / "scripts/exogenous_forecast/Exchange/TimeXer.sh",
    ROOT / "scripts/exogenous_forecast/Weather/TimeXer.sh",
]


def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def parse_supported_flags():
    supported = set()
    for line in RUN_PY.read_text().splitlines():
        line = line.strip()
        if "add_argument('--" not in line:
            continue
        start = line.index("add_argument('--") + len("add_argument('--")
        end = line.index("'", start)
        supported.add(line[start:end])
    return supported


SUPPORTED_FLAGS = parse_supported_flags()


def normalize_line(raw_line):
    line = raw_line.rstrip("\n")
    stripped = line.lstrip()
    if stripped.startswith("#"):
        stripped = stripped[1:].lstrip()
    return stripped


ASSIGNMENT_RE = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
SHELL_VAR_RE = re.compile(r"\$(\w+)|\$\{([^}]+)\}")


def parse_assignment(line):
    match = ASSIGNMENT_RE.match(line)
    if not match:
        return None
    key = match.group(1)
    value = match.group(2).strip()
    if not value:
        return key, ""
    try:
        pieces = shlex.split(value)
        parsed = " ".join(pieces) if pieces else value.strip("'\"")
    except ValueError:
        parsed = value.strip("'\"")
    return key, parsed


def substitute_shell_vars(line, variables):
    def replace(match):
        key = match.group(1) or match.group(2)
        return variables.get(key, match.group(0))

    return SHELL_VAR_RE.sub(replace, line)


def parse_script_commands(script_path):
    commands = []
    current = []
    variables = {}
    for raw_line in script_path.read_text().splitlines():
        line = normalize_line(raw_line)
        assignment = parse_assignment(line)
        if assignment and not current:
            key, value = assignment
            variables[key] = value
            continue
        line = substitute_shell_vars(line, variables)
        if not current:
            if line.startswith("python") and "run.py" in line:
                current.append(line)
                if not line.rstrip().endswith("\\"):
                    commands.append(" ".join(current))
                    current = []
        else:
            if not line:
                continue
            current.append(line)
            if not line.rstrip().endswith("\\"):
                commands.append(" ".join(current))
                current = []
    return commands


def command_to_args(command):
    cleaned = command.replace("\\", " ")
    tokens = shlex.split(cleaned)
    args = {}
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token.startswith("--"):
            key = token[2:]
            if idx + 1 < len(tokens) and not tokens[idx + 1].startswith("--"):
                args[key] = tokens[idx + 1]
                idx += 2
            else:
                args[key] = True
                idx += 1
        else:
            idx += 1
    return args


def sanitize_args(args):
    cleaned = {}
    for key, value in args.items():
        if key in SUPPORTED_FLAGS:
            cleaned[key] = value
    for key, value in RUN_DEFAULTS.items():
        cleaned.setdefault(key, value)
    cleaned["is_training"] = str(cleaned.get("is_training", "1"))
    cleaned["itr"] = str(cleaned.get("itr", "1"))
    return cleaned


def infer_dataset(args):
    data = args.get("data", "")
    data_path = args.get("data_path", "")
    model_id = args.get("model_id", "")
    if data == "Futures":
        stem = Path(data_path).stem.replace("futures_", "")
        return f"Futures_{stem}"
    if model_id.startswith("ili_") or "illness" in args.get("root_path", ""):
        return "ILI"
    if data in {"ETTh1", "ETTh2", "ETTm1", "ETTm2"}:
        return data
    if "exchange_rate" in args.get("root_path", "") or model_id.startswith("Exchange"):
        return "Exchange"
    if "weather" in args.get("root_path", "") or model_id.startswith("weather"):
        return "Weather"
    return model_id


def model_alias(model_name):
    return "EATA" if model_name in {"EATA", "TD_CaA"} else model_name


def make_job(args, source, variant="seed", notes=""):
    args = sanitize_args(args)
    dataset = infer_dataset(args)
    return {
        "source": source,
        "variant": variant,
        "notes": notes,
        "dataset": dataset,
        "pred_len": int(args["pred_len"]),
        "task_type": args["features"],
        "model": model_alias(args["model"]),
        "args": args,
    }


def manual_ms_jobs():
    jobs = []
    ili_eata = [
        {"pred_len": 24, "dropout": "0.05", "batch_size": "32", "d_model": "64", "learning_rate": "0.05"},
        {"pred_len": 36, "dropout": "0.15", "batch_size": "16", "d_model": "64", "learning_rate": "0.01"},
        {"pred_len": 48, "dropout": "0.25", "batch_size": "12", "d_model": "64", "learning_rate": "0.005"},
        {"pred_len": 60, "dropout": "0.15", "batch_size": "16", "d_model": "64", "learning_rate": "0.001"},
    ]
    for item in ili_eata:
        args = {
            "model": "EATA",
            "task_name": "long_term_forecast",
            "is_training": "1",
            "root_path": "./dataset/illness/",
            "data_path": "national_illness.csv",
            "model_id": f"ili_36_{item['pred_len']}",
            "data": "custom",
            "target": "OT",
            "features": "MS",
            "seq_len": "36",
            "label_len": "18",
            "pred_len": str(item["pred_len"]),
            "e_layers": "4",
            "d_layers": "1",
            "factor": "3",
            "enc_in": "7",
            "dec_in": "7",
            "c_out": "1",
            "des": "EATA-MS",
            "dropout": item["dropout"],
            "batch_size": item["batch_size"],
            "d_model": item["d_model"],
            "k_lookback": "36",
            "learning_rate": item["learning_rate"],
            "train_epochs": "10",
            "patience": "3",
            "method": "Dynamic",
            "hidden": "28",
            "bias": True,
            "interact": True,
            "itr": "1",
        }
        jobs.append(make_job(args, "manual:exogenous_forecast/ILI/EATA.sh"))

    ili_timexer = [
        {"pred_len": 24, "e_layers": "1", "d_model": "256", "d_ff": "1024", "batch_size": "16"},
        {"pred_len": 36, "e_layers": "1", "d_model": "256", "d_ff": "1024", "batch_size": "16"},
        {"pred_len": 48, "e_layers": "2", "d_model": "512", "d_ff": "1024", "batch_size": "4"},
        {"pred_len": 60, "e_layers": "2", "d_model": "256", "d_ff": "1024", "batch_size": "16"},
    ]
    for item in ili_timexer:
        args = {
            "model": "TimeXer",
            "task_name": "long_term_forecast",
            "is_training": "1",
            "root_path": "./dataset/illness/",
            "data_path": "national_illness.csv",
            "model_id": f"ili_36_{item['pred_len']}",
            "data": "custom",
            "target": "OT",
            "features": "MS",
            "seq_len": "36",
            "label_len": "18",
            "pred_len": str(item["pred_len"]),
            "e_layers": item["e_layers"],
            "factor": "3",
            "enc_in": "7",
            "dec_in": "7",
            "c_out": "1",
            "des": "TimeXer-MS",
            "d_model": item["d_model"],
            "d_ff": item["d_ff"],
            "batch_size": item["batch_size"],
            "itr": "1",
        }
        jobs.append(make_job(args, "manual:exogenous_forecast/ILI/TimeXer.sh"))

    for future_name in ["TA", "RB", "M"]:
        for pred_len, dropout, lr in [(96, "0.2", "0.005"), (192, "0.05", "0.00075"), (336, "0.05", "0.001"), (720, "0.05", "0.001")]:
            args = {
                "model": "EATA",
                "task_name": "long_term_forecast",
                "is_training": "1",
                "root_path": "./dataset/futures/",
                "data_path": f"futures_{future_name}.csv",
                "model_id": f"Futures_{future_name}_96_{pred_len}",
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
                "des": "EATA-MS",
                "dropout": dropout,
                "d_model": "64",
                "k_lookback": "64",
                "batch_size": "32",
                "learning_rate": lr,
                "train_epochs": "20",
                "patience": "3",
                "method": "Dynamic",
                "hidden": "28",
                "bias": True,
                "interact": True,
                "itr": "1",
            }
            jobs.append(make_job(args, f"manual:exogenous_forecast/Futures/EATA_{future_name}.sh"))

            tx_args = {
                "model": "TimeXer",
                "task_name": "long_term_forecast",
                "is_training": "1",
                "root_path": "./dataset/futures/",
                "data_path": f"futures_{future_name}.csv",
                "model_id": f"futures_{future_name}_96_{pred_len}",
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
                "des": "TimeXer-MS",
                "d_model": "128",
                "d_ff": "512",
                "batch_size": "32",
                "itr": "1",
            }
            jobs.append(make_job(tx_args, f"manual:exogenous_forecast/Futures/TimeXer_{future_name}.sh"))
    return jobs


def manual_m_jobs():
    jobs = []
    for future_name in ["RB", "M"]:
        for pred_len, dropout, d_model, lr in [(96, "0.2", "64", "0.005"), (192, "0.05", "64", "0.00075"), (336, "0.05", "64", "0.001"), (720, "0.05", "64", "0.001")]:
            args = {
                "model": "EATA",
                "task_name": "long_term_forecast",
                "is_training": "1",
                "root_path": "./dataset/futures/",
                "data_path": f"futures_{future_name}.csv",
                "model_id": f"Futures_{future_name}_96_{pred_len}",
                "data": "Futures",
                "target": "LastPrice",
                "features": "M",
                "seq_len": "96",
                "label_len": "48",
                "pred_len": str(pred_len),
                "e_layers": "1",
                "factor": "3",
                "enc_in": "12",
                "dec_in": "12",
                "c_out": "12",
                "des": "Exp",
                "dropout": dropout,
                "moving_avg": "10",
                "batch_size": "32",
                "learning_rate": lr,
                "train_epochs": "20",
                "k_lookback": "64",
                "patience": "3",
                "itr": "1",
                "method": "Dynamic",
                "hidden": "28",
                "bias": True,
                "interact": True,
                "d_model": d_model,
            }
            jobs.append(make_job(args, f"manual:long_term_forecast/Futures_script/EATA_{future_name}.sh"))
    return jobs


def seed_jobs():
    jobs = []
    for script_path in EXPLICIT_EATA_M_SCRIPTS + EXPLICIT_EATA_MS_SCRIPTS + EXPLICIT_TIMEXER_MS_SCRIPTS:
        for command in parse_script_commands(script_path):
            args = sanitize_args(command_to_args(command))
            model = model_alias(args.get("model", ""))
            if model not in {"EATA", "TimeXer"}:
                continue
            jobs.append(make_job(args, str(script_path.relative_to(ROOT))))
    jobs.extend(manual_m_jobs())
    jobs.extend(manual_ms_jobs())
    return dedupe_jobs(jobs)


def dedupe_jobs(jobs):
    deduped = {}
    for job in jobs:
        deduped[setting_from_args(job["args"])] = job
    return list(deduped.values())


def next_d_model(base):
    choices = [32, 64, 128, 256, 512]
    for choice in choices:
        if choice > base:
            return choice
    return base


def tune_variants(job):
    if job["model"] != "EATA":
        return [job]

    args = deepcopy(job["args"])
    seq_len = int(args["seq_len"])
    base_dm = int(args["d_model"])
    base_dp = float(args.get("dropout", "0.1"))
    base_lr = float(args["learning_rate"])
    base_bs = int(args["batch_size"])
    base_lb = int(args.get("k_lookback", seq_len))
    dataset = job["dataset"]
    pred_len = int(args["pred_len"])

    if dataset == "Weather":
        batch_cap = 8
    elif dataset == "Exchange":
        batch_cap = 32
    elif dataset.startswith("Futures"):
        batch_cap = 32
    elif dataset == "ILI":
        batch_cap = 32
    else:
        batch_cap = 128

    candidates = []
    candidates.append((job, "seed"))

    tuned = deepcopy(job)
    tuned["variant"] = "seq_lookback"
    tuned["args"] = deepcopy(args)
    tuned["args"]["k_lookback"] = str(seq_len)
    tuned["args"]["d_model"] = str(max(64, min(256, base_dm)))
    tuned["args"]["dropout"] = f"{max(0.05, round(base_dp - 0.05, 2)):.2f}"
    candidates.append((tuned, tuned["variant"]))

    stable = deepcopy(job)
    stable["variant"] = "stable_lr"
    stable["args"] = deepcopy(args)
    stable["args"]["learning_rate"] = f"{max(base_lr * 0.5, 1e-5):.6f}".rstrip("0").rstrip(".")
    stable["args"]["batch_size"] = str(min(batch_cap, max(base_bs, base_bs * 2 if base_bs < 16 else base_bs)))
    stable["args"]["dropout"] = f"{min(0.25, round(base_dp + 0.05, 2)):.2f}"
    stable["args"]["k_lookback"] = str(seq_len if seq_len <= 96 else base_lb)
    candidates.append((stable, stable["variant"]))

    wider = deepcopy(job)
    wider["variant"] = "wider_model"
    wider["args"] = deepcopy(args)
    wider["args"]["d_model"] = str(next_d_model(base_dm))
    wider["args"]["batch_size"] = str(max(4, base_bs // 2))
    wider["args"]["learning_rate"] = f"{max(base_lr * 0.75, 1e-5):.6f}".rstrip("0").rstrip(".")
    wider["args"]["k_lookback"] = str(max(base_lb, min(seq_len, 64 if seq_len >= 64 else seq_len)))
    candidates.append((wider, wider["variant"]))

    if pred_len >= seq_len:
        longh = deepcopy(job)
        longh["variant"] = "long_horizon"
        longh["args"] = deepcopy(args)
        longh["args"]["dropout"] = f"{max(0.1, min(0.25, round(base_dp + 0.1, 2))):.2f}"
        longh["args"]["learning_rate"] = f"{max(base_lr * 0.5, 5e-5):.6f}".rstrip("0").rstrip(".")
        longh["args"]["batch_size"] = str(max(4, base_bs // 2))
        longh["args"]["k_lookback"] = str(seq_len)
        candidates.append((longh, longh["variant"]))

    seen = {}
    for candidate, variant in candidates:
        setting = setting_from_args(candidate["args"])
        if setting not in seen:
            seen[setting] = candidate
    return list(seen.values())


def build_queue():
    seeds = seed_jobs()
    queue = []
    for job in seeds:
        if job["model"] == "TimeXer":
            queue.append(job)
        else:
            queue.extend(tune_variants(job))
    return prioritize_jobs(dedupe_jobs(queue))


def prioritize_jobs(jobs):
    def score(job):
        dataset = job["dataset"]
        pred_len = job["pred_len"]
        task_type = job["task_type"]
        targets = M_TARGETS if task_type == "M" else MS_TARGETS
        base = 1 if job["model"] == "TimeXer" and task_type == "MS" else 2
        if dataset in targets and pred_len in targets[dataset]:
            base = 0
        if dataset == "ILI":
            base -= 1
        if job["variant"] == "seed":
            base -= 0.2
        return base, dataset, pred_len, job["variant"]
    return sorted(jobs, key=score)


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


def existing_completed_settings():
    completed = set()
    if RUNS_PATH.exists():
        for line in RUNS_PATH.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") == "completed":
                completed.add(record["setting"])
    return completed


def append_run_record(record):
    with RUNS_PATH.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def load_run_records():
    records = []
    if not RUNS_PATH.exists():
        return records
    for line in RUNS_PATH.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def backfill_ili_baselines():
    records = []
    patterns = [
        ("TimeXer", 24),
        ("TimeXer", 36),
        ("TimeXer", 48),
        ("TimeXer", 60),
        ("TimeMixer", 24),
        ("TimeMixer", 36),
        ("TimeMixer", 48),
        ("TimeMixer", 60),
        ("iTransformer", 24),
        ("iTransformer", 36),
        ("iTransformer", 48),
        ("iTransformer", 60),
    ]
    for model, pred_len in patterns:
        matches = sorted(RESULTS_DIR.glob(f"long_term_forecast_ili_36_{pred_len}_{model}_*"))
        if model == "TimeXer":
            # Keep the M baseline table from accidentally picking later MS runs.
            matches = [item for item in matches if item.name.endswith("_Exp")]
        if not matches:
            continue
        metrics = result_metrics(matches[-1].name)
        if metrics is None:
            continue
        records.append(
            {
                "status": "completed",
                "setting": matches[-1].name,
                "dataset": "ILI",
                "task_type": "M",
                "pred_len": pred_len,
                "model": model,
                "metrics": metrics,
                "source": "backfill:results",
            }
        )
    return records


def best_by_key(records):
    best = {}
    for record in records:
        if record.get("status") != "completed":
            continue
        key = (record["task_type"], record["dataset"], int(record["pred_len"]), record["model"])
        metrics = record.get("metrics")
        if not metrics:
            continue
        if key not in best:
            best[key] = record
            continue
        cur = best[key]["metrics"]
        if metrics["mse"] < cur["mse"] or (
            abs(metrics["mse"] - cur["mse"]) < 1e-12 and metrics["mae"] < cur["mae"]
        ):
            best[key] = record
    return best


def format_metric_pair(pair):
    if pair is None:
        return "-", "-"
    return f"{pair[0]:.3f}", f"{pair[1]:.3f}"


def format_completed_metrics(record):
    if not record:
        return "-", "-"
    return f"{record['metrics']['mse']:.3f}", f"{record['metrics']['mae']:.3f}"


def param_summary(record):
    if not record or "args" not in record:
        return "-"
    args = record["args"]
    return (
        f"d_model={args.get('d_model')} "
        f"dropout={args.get('dropout')} "
        f"k_lookback={args.get('k_lookback')} "
        f"lr={args.get('learning_rate')} "
        f"bs={args.get('batch_size')}"
    )


def write_state(running_jobs, pending_count):
    state = {
        "updated_at": now_utc(),
        "pending_jobs": pending_count,
        "running_jobs": running_jobs,
    }
    STATE_PATH.write_text(json.dumps(state, indent=2))


def write_report():
    run_records = load_run_records()
    ili_baseline_records = backfill_ili_baselines()
    combined = ili_baseline_records + run_records
    best = best_by_key(combined)
    ili_best = best_by_key(ili_baseline_records)
    state = {"updated_at": now_utc(), "pending_jobs": 0, "running_jobs": []}
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text())
    else:
        completed_settings = existing_completed_settings()
        pending_jobs = 0
        for job in build_queue():
            setting = setting_from_args(job["args"])
            if setting in completed_settings or result_metrics(setting) is not None:
                continue
            pending_jobs += 1
        state["pending_jobs"] = pending_jobs

    lines = []
    lines.append("# EATA Study Report")
    lines.append("")
    lines.append(f"Updated: {state.get('updated_at', now_utc())}")
    lines.append(f"Pending jobs: {state.get('pending_jobs', 0)}")
    lines.append(f"Running jobs: {len(state.get('running_jobs', []))}")
    lines.append("")

    lines.append("## ILI Baselines (M)")
    lines.append("")
    lines.append("| Horizon | TimeXer MSE | TimeXer MAE | TimeMixer MSE | TimeMixer MAE | iTransformer MSE | iTransformer MAE |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for horizon in REQUESTED_M["ILI"]:
        tx = ili_best.get(("M", "ILI", horizon, "TimeXer"))
        tm = ili_best.get(("M", "ILI", horizon, "TimeMixer"))
        it = ili_best.get(("M", "ILI", horizon, "iTransformer"))
        tx_mse, tx_mae = format_completed_metrics(tx)
        tm_mse, tm_mae = format_completed_metrics(tm)
        it_mse, it_mae = format_completed_metrics(it)
        lines.append(f"| {horizon} | {tx_mse} | {tx_mae} | {tm_mse} | {tm_mae} | {it_mse} | {it_mae} |")
    lines.append("")

    lines.append("## TimeXer Baselines (MS)")
    lines.append("")
    lines.append("| Dataset | Horizon | TimeXer MSE | TimeXer MAE |")
    lines.append("| --- | ---: | ---: | ---: |")
    for dataset, horizons in REQUESTED_MS.items():
        for horizon in horizons:
            record = best.get(("MS", dataset, horizon, "TimeXer"))
            mse, mae = format_completed_metrics(record)
            lines.append(f"| {dataset} | {horizon} | {mse} | {mae} |")
    lines.append("")

    lines.append("## EATA Best So Far (M)")
    lines.append("")
    lines.append("| Dataset | Horizon | Previous MSE | Previous MAE | Best Study MSE | Best Study MAE | Params |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for dataset, horizons in REQUESTED_M.items():
        if dataset == "ILI":
            target_dict = {}
        else:
            target_dict = M_TARGETS.get(dataset, {})
        for horizon in horizons:
            prev_mse, prev_mae = format_metric_pair(target_dict.get(horizon))
            record = best.get(("M", dataset, horizon, "EATA"))
            best_mse, best_mae = format_completed_metrics(record)
            lines.append(
                f"| {dataset} | {horizon} | {prev_mse} | {prev_mae} | {best_mse} | {best_mae} | {param_summary(record)} |"
            )
    lines.append("")

    lines.append("## EATA Best So Far (MS)")
    lines.append("")
    lines.append("| Dataset | Horizon | Previous MSE | Previous MAE | Best Study MSE | Best Study MAE | Params |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for dataset, horizons in REQUESTED_MS.items():
        target_dict = MS_TARGETS.get(dataset, {})
        for horizon in horizons:
            prev_mse, prev_mae = format_metric_pair(target_dict.get(horizon))
            record = best.get(("MS", dataset, horizon, "EATA"))
            best_mse, best_mae = format_completed_metrics(record)
            lines.append(
                f"| {dataset} | {horizon} | {prev_mse} | {prev_mae} | {best_mse} | {best_mae} | {param_summary(record)} |"
            )
    lines.append("")

    lines.append("## Recent Completed Runs")
    lines.append("")
    lines.append("| Time | Task | Dataset | Horizon | Model | MSE | MAE | Variant | Setting |")
    lines.append("| --- | --- | --- | ---: | --- | ---: | ---: | --- | --- |")
    completed = [r for r in run_records if r.get("status") == "completed"][-30:]
    for record in reversed(completed):
        lines.append(
            f"| {record.get('finished_at', '-')} | {record['task_type']} | {record['dataset']} | {record['pred_len']} | "
            f"{record['model']} | {record['metrics']['mse']:.3f} | {record['metrics']['mae']:.3f} | "
            f"{record.get('variant', '-')} | `{record['setting']}` |"
        )
    lines.append("")

    if state.get("running_jobs"):
        lines.append("## Running Jobs")
        lines.append("")
        lines.append("| GPU | Task | Dataset | Horizon | Model | Variant | Setting |")
        lines.append("| --- | --- | --- | ---: | --- | --- | --- |")
        for item in state["running_jobs"]:
            lines.append(
                f"| {item['gpu']} | {item['task_type']} | {item['dataset']} | {item['pred_len']} | "
                f"{item['model']} | {item['variant']} | `{item['setting']}` |"
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


def append_completion_record(job, status, metrics, returncode, log_path="", notes=None):
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
        "notes": job["notes"] if notes is None else notes,
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


def running_payload(running):
    return [
        {
            "gpu": gpu,
            "dataset": item["job"]["dataset"],
            "task_type": item["job"]["task_type"],
            "pred_len": item["job"]["pred_len"],
            "model": item["job"]["model"],
            "variant": item["job"]["variant"],
            "setting": item["setting"],
        }
        for gpu, item in running.items()
    ]


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
        if job is None:
            continue
        if not job_process_alive(job):
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


def launch_queue(gpu_ids, limit=None, poll_secs=20):
    LOG_DIR.mkdir(exist_ok=True)
    queue = build_queue()
    completed_settings = existing_completed_settings()
    running = restore_running_jobs(queue, completed_settings)
    reserved_settings = {item["setting"] for item in running.values()}
    pending = []
    for job in queue:
        setting = setting_from_args(job["args"])
        if setting in completed_settings or setting in reserved_settings:
            continue
        if result_metrics(setting) is not None:
            if setting not in completed_settings:
                append_completion_record(job, "completed", result_metrics(setting), 0, "", "backfill:existing_result")
                completed_settings.add(setting)
            continue
        pending.append(job)
    if limit is not None:
        pending = pending[:limit]

    write_report()
    while pending or running:
        launched = False
        while pending and len(running) < len(gpu_ids):
            available = [gpu for gpu in gpu_ids if gpu not in running]
            if not available:
                break
            gpu = available[0]
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
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=ROOT,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
            except BlockingIOError:
                log_file.close()
                pending.insert(0, job)
                break
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

    write_report()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["report", "launch", "list"], default="report")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--poll-secs", type=int, default=20)
    args = parser.parse_args()

    if args.mode == "report":
        write_report()
        print(f"Wrote {REPORT_PATH}")
        return

    if args.mode == "list":
        jobs = build_queue()
        for job in jobs[: args.limit]:
            print(job["task_type"], job["dataset"], job["pred_len"], job["model"], job["variant"], setting_from_args(job["args"]))
        print(f"total_jobs={len(jobs)}")
        return

    gpu_ids = [int(item) for item in args.gpus.split(",") if item.strip()]
    launch_queue(gpu_ids, limit=args.limit, poll_secs=args.poll_secs)


if __name__ == "__main__":
    main()
