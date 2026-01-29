import os
import subprocess
import itertools
import random
import re
import threading
import queue
import time
import json
from datetime import datetime

# ================= 配置区域 =================

GPU_IDS = [0, 1, 2] # 你的三张卡
RESULT_FILE = "search_log.csv"
file_lock = threading.Lock()

# 限制每个任务最多尝试次数（防止无限跑，设为 9999 则几乎等于跑完所有）
MAX_TRIALS = 1000 

# 搜索空间
param_grid = {
    'd_model': [32, 64, 128, 256],
    'dropout': [0.05, 0.1, 0.15, 0.2, 0.25],
    'batch_size': [16, 32, 64],
    'learning_rate': [0.05, 0.03, 0.01, 0.005, 0.003, 0.001, 0.0005, 0.0003, 0.0001],
    'k_lookback': [32, 64, 96],
    'hidden': [16, 24, 28, 32, 48, 64],
}

# 命令模板
base_cmd_template = (
    "python run.py "
    "--task_name long_term_forecast "
    "--is_training 1 "
    "--root_path ./dataset/{dir} "
    "--data_path {data_path} "
    "--model_id {data_path}_{seq_len}_{pred_len} "
    "--model {model_name} "
    "--data {data_name} "
    "--features M "
    "--seq_len {seq_len} "
    "--label_len 48 "
    "--pred_len {pred_len} "
    "--e_layers 2 "
    "--d_layers 1 "
    "--factor 3 "
    "--enc_in {dim} "
    "--dec_in {dim} "
    "--c_out {dim} "
    "--d_ff 128 "
    "--des 'Exp' "
    "--itr 1 "
    "--train_epochs 20 "
    "--patience 3 "
    "--method 'Dynamic' "
    "--bias "
    "--interact "
    "--d_model {d_model} "
    "--dropout {dropout} "
    "--batch_size {batch_size} "
    "--learning_rate {learning_rate} "
    "--k_lookback {k_lookback} "
    "--hidden {hidden} "
)

target_tasks = [
     {
        "dir": "ETT-small/",
        "data_path": "ETTm2.csv",
        "data_name": "ETTm2",
        "model_name": "TD_CaA",
        "seq_len": 96,
        "pred_len": 96,
        "dim": 7,
        "target_mse": 0.171121642
    },
     {
        "dir": "ETT-small/",
        "data_path": "ETTm2.csv",
        "data_name": "ETTm2",
        "model_name": "TD_CaA",
        "seq_len": 96,
        "pred_len": 192,
        "dim": 7,
        "target_mse": 0.236761913
    },
     {
        "dir": "ETT-small/",
        "data_path": "ETTm2.csv",
        "data_name": "ETTm2",
        "model_name": "TD_CaA",
        "seq_len": 96,
        "pred_len": 336,
        "dim": 7,
        "target_mse": 0.297930926
    },
     {
        "dir": "ETT-small/",
        "data_path": "ETTm2.csv",
        "data_name": "ETTm2",
        "model_name": "TD_CaA",
        "seq_len": 96,
        "pred_len": 720,
        "dim": 7,
        "target_mse": 0.39240694
    },
]

# ================= 辅助函数 =================

def get_combinations(grid):
    keys = grid.keys()
    values = grid.values()
    return [dict(zip(keys, v)) for v in itertools.product(*values)]

def parse_output(output):
    """解析日志，提取 mse 和 mae"""
    mse_match = re.search(r"mse:(\d+\.?\d*)", output)
    mae_match = re.search(r"mae:(\d+\.?\d*)", output)
    mse = float(mse_match.group(1)) if mse_match else float('inf')
    mae = float(mae_match.group(1)) if mae_match else float('inf')
    return mse, mae

def save_record(record_type, task, mse, mae, params):
    """写入 CSV，记录这一刻的突破"""
    with file_lock:
        timestamp = datetime.now().strftime("%H:%M:%S")
        # 将 params 字典转为 json 字符串，避免逗号干扰 csv 格式
        params_str = json.dumps(params)
        
        line = f"{timestamp},{record_type},{task['data_name']},{task['pred_len']},{mse},{mae},{params_str}\n"
        
        # 打印到屏幕
        print(f"[{record_type}] {task['data_name']}-{task['pred_len']} | MSE:{mse} MAE:{mae}")
        
        with open(RESULT_FILE, "a") as f:
            f.write(line)

def run_cmd(gpu_id, cmd):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        # 运行命令，超时时间设为15分钟防止卡死
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env, timeout=1200)
        return res.stdout + res.stderr
    except subprocess.TimeoutExpired:
        print(f"[GPU {gpu_id}] Timeout!")
        return ""
    except Exception as e:
        print(f"[GPU {gpu_id}] Error: {e}")
        return ""

# =================主要工作逻辑 =================

def worker(gpu_id, queue_obj):
    while True:
        try:
            task = queue_obj.get(timeout=3)
        except queue.Empty:
            break
        
        print(f"\n>>> GPU {gpu_id} 处理任务: {task['data_name']} len={task['pred_len']} (Target: {task['target_mse']})")
        
        # 1. 生成空间并打乱
        combos = get_combinations(param_grid)
        random.shuffle(combos)
        
        # 2. 初始化该任务的擂台记录
        current_best_mse = float('inf')
        current_best_mae = float('inf')
        
        # 3. 开始搜索
        for idx, params in enumerate(combos):
            if idx >= MAX_TRIALS:
                print(f"[GPU {gpu_id}] 达到最大尝试次数，停止任务")
                break
                
            # 拼装参数 & 运行
            cmd_params = {**task, **params}
            # 简单的进度展示
            if idx % 10 == 0:
                print(f"[GPU {gpu_id}] Progress {idx}/{len(combos)} | Best MSE: {current_best_mse:.4f}")

            cmd = base_cmd_template.format(**cmd_params)
            log_output = run_cmd(gpu_id, cmd)
            mse, mae = parse_output(log_output)
            
            # 如果跑崩了(inf)，直接跳过
            if mse == float('inf'):
                continue
            
            # === 核心逻辑：擂台比较 ===
            updated = False
            
            # 1. 检查是否破了 MSE 记录
            if mse < current_best_mse:
                current_best_mse = mse
                save_record("NEW_BEST_MSE", task, mse, mae, params)
                updated = True
                
            # 2. 检查是否破了 MAE 记录 (如果参数不一样，且MAE确实更低)
            if mae < current_best_mae:
                current_best_mae = mae
                # 如果刚才已经存过 NEW_BEST_MSE，且这次只是顺便更新了 MAE，不必重复存文件
                # 除非你希望显式记录 "这个参数 MAE 最好"
                if not updated: 
                    save_record("NEW_BEST_MAE", task, mse, mae, params)
            
            # 3. 检查是否达标 (低于 Target)
            if mse < task['target_mse']:
                save_record("TARGET_HIT", task, mse, mae, params)
                print(f"*** GPU {gpu_id} 任务达标，提前结束！ ***")
                break # 直接退出当前任务的 for 循环，去领下一个任务
        
        queue_obj.task_done()

# ================= 启动代码 =================

def main():
    # 初始化文件头
    if not os.path.exists(RESULT_FILE):
        with open(RESULT_FILE, 'w') as f:
            f.write("Time,RecordType,Data,PredLen,MSE,MAE,Params_JSON\n")
            
    # 填充队列
    q = queue.Queue()
    for t in target_tasks:
        q.put(t)
        
    print(f"Start Tuning with {len(GPU_IDS)} GPUs for {q.qsize()} tasks...")
    
    threads = []
    for gid in GPU_IDS:
        t = threading.Thread(target=worker, args=(gid, q))
        t.start()
        threads.append(t)
        time.sleep(5) # 错峰启动
        
    for t in threads:
        t.join()
    print("Done.")

if __name__ == "__main__":
    main()
