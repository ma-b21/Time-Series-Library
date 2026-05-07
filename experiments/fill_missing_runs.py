#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import eata_study as s


PRIMARY_TARGETS = {
    ("TimeXer", "MS", "ETTm1"): [96, 192, 336, 720],
    ("TimeXer", "MS", "ETTm2"): [96, 192, 336, 720],
    ("TimeXer", "MS", "Futures_TA"): [96, 192, 336, 720],
    ("TimeXer", "MS", "Futures_RB"): [96, 192, 336, 720],
    ("EATA", "M", "ILI"): [24, 36, 48, 60],
    ("EATA", "MS", "ETTm1"): [96, 192, 336, 720],
    ("EATA", "MS", "ETTm2"): [96, 192, 336, 720],
    ("EATA", "MS", "ILI"): [24, 36, 48, 60],
    ("EATA", "MS", "Futures_TA"): [96, 192, 336, 720],
    ("EATA", "MS", "Futures_RB"): [96, 192, 336, 720],
}

EXTRA_TARGETS = {
    ("EATA", "MS", "ETTh1"): [720],
    ("EATA", "MS", "Exchange"): [336, 720],
}

VARIANT_PRIORITY = {
    "seed": 0,
    "seq_lookback": 1,
    "wider_model": 2,
    "stable_lr": 3,
    "long_horizon": 4,
}

GROUP_PRIORITY = {
    ("TimeXer", "MS", "ETTm1"): 0,
    ("TimeXer", "MS", "ETTm2"): 1,
    ("TimeXer", "MS", "Futures_TA"): 2,
    ("TimeXer", "MS", "Futures_RB"): 3,
    ("EATA", "M", "ILI"): 4,
    ("EATA", "MS", "ILI"): 5,
    ("EATA", "MS", "ETTm1"): 6,
    ("EATA", "MS", "ETTm2"): 7,
    ("EATA", "MS", "Futures_TA"): 8,
    ("EATA", "MS", "Futures_RB"): 9,
    ("EATA", "MS", "ETTh1"): 10,
    ("EATA", "MS", "Exchange"): 11,
}


def target_map(include_extra):
    targets = dict(PRIMARY_TARGETS)
    if include_extra:
        targets.update(EXTRA_TARGETS)
    return targets


def current_best():
    records = s.backfill_ili_baselines() + s.load_run_records()
    return s.best_by_key(records)


def missing_tuples(include_extra):
    best = current_best()
    missing = []
    for key, horizons in target_map(include_extra).items():
        model, task_type, dataset = key
        for pred_len in horizons:
            if (task_type, dataset, pred_len, model) not in best:
                missing.append((model, task_type, dataset, pred_len))
    return missing


def manual_ili_m_seed_jobs():
    configs = [
        {"pred_len": 24, "dropout": "0.05", "batch_size": "32", "d_model": "64", "learning_rate": "0.05"},
        {"pred_len": 36, "dropout": "0.15", "batch_size": "16", "d_model": "64", "learning_rate": "0.01"},
        {"pred_len": 48, "dropout": "0.25", "batch_size": "12", "d_model": "64", "learning_rate": "0.005"},
        {"pred_len": 60, "dropout": "0.15", "batch_size": "16", "d_model": "64", "learning_rate": "0.001"},
    ]
    jobs = []
    for item in configs:
        args = {
            "model": "EATA",
            "task_name": "long_term_forecast",
            "is_training": "1",
            "root_path": "./dataset/illness/",
            "data_path": "national_illness.csv",
            "model_id": f"ili_36_{item['pred_len']}",
            "data": "custom",
            "target": "OT",
            "features": "M",
            "seq_len": "36",
            "label_len": "18",
            "pred_len": str(item["pred_len"]),
            "e_layers": "4",
            "d_layers": "1",
            "factor": "3",
            "enc_in": "7",
            "dec_in": "7",
            "c_out": "7",
            "des": "Exp",
            "dropout": item["dropout"],
            "batch_size": item["batch_size"],
            "d_model": item["d_model"],
            "moving_avg": "10",
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
        jobs.append(s.make_job(args, "manual:missing/ILI_M.sh"))
    return jobs


def candidate_pool():
    pool = []
    for job in s.build_queue():
        if (job["model"], job["task_type"], job["dataset"]) == ("EATA", "M", "ILI"):
            continue
        pool.append(job)
    for job in manual_ili_m_seed_jobs():
        pool.extend(s.tune_variants(job))
    return s.dedupe_jobs(pool)


def job_key(job):
    return job["model"], job["task_type"], job["dataset"], job["pred_len"]


def sort_key(job):
    group = (job["model"], job["task_type"], job["dataset"])
    return (
        GROUP_PRIORITY.get(group, 999),
        job["pred_len"],
        VARIANT_PRIORITY.get(job["variant"], 99),
        s.setting_from_args(job["args"]),
    )


def select_jobs(include_extra=False, max_eata_variants=4):
    missing = set(missing_tuples(include_extra))
    grouped = defaultdict(list)
    for job in candidate_pool():
        key = job_key(job)
        if key in missing:
            grouped[key].append(job)

    selected = []
    for key in sorted(missing, key=lambda item: (GROUP_PRIORITY.get(item[:3], 999), item[3], item[0])):
        jobs = sorted(grouped.get(key, []), key=sort_key)
        if key[0] == "TimeXer":
            selected.extend(jobs[:1])
            continue
        chosen = []
        seen_variants = set()
        for job in jobs:
            variant = job["variant"]
            if variant in seen_variants:
                continue
            chosen.append(job)
            seen_variants.add(variant)
            if len(chosen) >= max_eata_variants:
                break
        selected.extend(chosen)
    return s.dedupe_jobs(selected), sorted(missing, key=lambda item: (GROUP_PRIORITY.get(item[:3], 999), item[3], item[0]))


def backfill_existing_results(jobs):
    completed_settings = s.existing_completed_settings()
    backfilled = 0
    for job in jobs:
        setting = s.setting_from_args(job["args"])
        if setting in completed_settings:
            continue
        metrics = s.result_metrics(setting)
        if metrics is None:
            continue
        s.append_completion_record(job, "completed", metrics, 0, "", "backfill:missing_focus_existing")
        completed_settings.add(setting)
        backfilled += 1
    if backfilled:
        s.write_report()
    return backfilled


def launch_jobs(jobs, gpu_ids, poll_secs=20, limit=None):
    s.LOG_DIR.mkdir(exist_ok=True)
    completed_settings = s.existing_completed_settings()
    running = s.restore_running_jobs(jobs, completed_settings)
    reserved_settings = {item["setting"] for item in running.values()}
    pending = []
    for job in jobs:
        setting = s.setting_from_args(job["args"])
        if setting in completed_settings or setting in reserved_settings:
            continue
        if s.result_metrics(setting) is not None:
            if setting not in completed_settings:
                s.append_completion_record(job, "completed", s.result_metrics(setting), 0, "", "backfill:missing_focus_existing")
                completed_settings.add(setting)
            continue
        pending.append(job)
    pending.sort(key=sort_key)
    if limit is not None:
        pending = pending[:limit]

    s.write_state(s.running_payload(running), len(pending))
    s.write_report()
    while pending or running:
        launched = False
        while pending and len(running) < len(gpu_ids):
            available = [gpu for gpu in gpu_ids if gpu not in running]
            if not available:
                break
            gpu = available[0]
            job = pending.pop(0)
            setting = s.setting_from_args(job["args"])
            log_path = s.LOG_DIR / f"{setting}.log"
            cmd = s.job_to_command(job["args"])
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
                    cwd=s.ROOT,
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
            s.write_state(s.running_payload(running), len(pending))
            s.write_report()

        finished = []
        for gpu, item in running.items():
            if item.get("external"):
                metrics = s.result_metrics(item["setting"])
                if metrics is not None:
                    s.append_completion_record(item["job"], "completed", metrics, 0, "", "backfill:restored_running")
                    finished.append(gpu)
                continue
            returncode = item["proc"].poll()
            if returncode is not None:
                item["log_file"].close()
                s.record_completed_job(item["job"], returncode, item["log_path"])
                finished.append(gpu)
        for gpu in finished:
            running.pop(gpu, None)
        s.write_state(s.running_payload(running), len(pending))
        if running:
            time.sleep(poll_secs)
        elif pending and not launched:
            time.sleep(min(poll_secs, 5))

    s.write_state([], 0)
    s.write_report()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["list", "launch"], default="launch")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6")
    parser.add_argument("--poll-secs", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-eata-variants", type=int, default=4)
    parser.add_argument("--include-extra-gaps", action="store_true")
    args = parser.parse_args()

    jobs, missing = select_jobs(include_extra=args.include_extra_gaps, max_eata_variants=args.max_eata_variants)
    backfilled = backfill_existing_results(jobs)
    jobs, missing = select_jobs(include_extra=args.include_extra_gaps, max_eata_variants=args.max_eata_variants)

    if args.mode == "list":
        print(f"backfilled={backfilled}")
        print(f"missing_tuples={len(missing)}")
        print(f"selected_jobs={len(jobs)}")
        for item in missing:
            print("MISSING", item)
        for job in jobs:
            print("JOB", job["model"], job["task_type"], job["dataset"], job["pred_len"], job["variant"], s.setting_from_args(job["args"]))
        return

    gpu_ids = [int(item) for item in args.gpus.split(",") if item.strip()]
    launch_jobs(jobs, gpu_ids, poll_secs=args.poll_secs, limit=args.limit)


if __name__ == "__main__":
    main()
