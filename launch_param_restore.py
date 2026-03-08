# launch_param_restore.py
# Batch launcher for param_restore.py
#
# Edit variables below and run:
#   python launch_param_restore.py

import os
import subprocess
from datetime import datetime


def main():
    # -----------------------------
    # You set these paths/values
    # -----------------------------
    python_bin = "python"  # or "python3"

    original_model_path = "/path/to/original_model"  # replace
    target_model_paths = [
        "/path/to/target_model_A",
        "/path/to/target_model_B",
    ]  # replace, target models to restore, need to share same architecture as original model
    output_root = "/path/to/output_root"  # replace
    
    # Restore settings
    k_percents = [1, 3, 5, 10, 20, 40, 60]
    region = "transformer"  # transformer/embedding/mlp/attention, the paper uses "transformer"
    strategy = "top"        # top/bottom/random, the paper uses "top"

    # Runtime
    diff_device = "cpu"     # cpu/auto/cuda/cuda:0... where to compute param diff + topk, the paper uses a gpu to compute these for 7B models. Note that model weights are still loaded on cpu to avoid OOM.
    dtype = "bf16"          # bf16/fp16/fp32, the paper uses "bf16" with gpu
    seed = 32

    # Optional log folder
    log_root = os.path.join(output_root, "_logs")
    os.makedirs(log_root, exist_ok=True)

    # -----------------------------
    # Launch loops
    # -----------------------------
    for tgt in target_model_paths:
        tgt_name = os.path.basename(tgt.rstrip("/"))
        for k in k_percents:
            out_dir = os.path.join(
                output_root,
                tgt_name,
                f"restore_{region}_{strategy}_{k}pct",
            )
            os.makedirs(out_dir, exist_ok=True)

            log_path = os.path.join(
                log_root,
                f"{tgt_name}_{region}_{strategy}_{k}pct_{datetime.now().strftime('%Y%m%d_%H%M')}.log",
            )

            cmd = [
                python_bin, "param_restore.py", "run",
                "--target_model_path", tgt,
                "--original_model_path", original_model_path,
                "--output_path", out_dir,
                "--k_percent", str(k),
                "--strategy", strategy,
                "--region", region,
                "--diff_device", diff_device,
                "--dtype", dtype,
                "--seed", str(seed),
                "--log_path", log_path,
                "--verbose", "True",
            ]

            print("\n=== Launch ===")
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
