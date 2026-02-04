import os
import re
import json
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM

import fire

EPS = 1e-8

def _in_region(name: str, region: str) -> bool:
    if region == "transformer":
        return ("layers" in name) and ("norm" not in name)
    if region == "embedding":
        return ("embed_tokens" in name) or ("lm_head" in name)
    if region == "attention":
        return "self_attn." in name
    if region == "mlp":
        return "mlp." in name
    if region in (None, "", "all"):
        return True
    raise ValueError(f"Unknown region: {region}")


def _layer_id(name: str) -> str:
    m = re.search(r"layers\.(\d+)", name)
    return m.group(1) if m else ""


@torch.no_grad()
def _relative_diff(old_t: torch.Tensor, new_t: torch.Tensor) -> torch.Tensor:
    # tensor_diff = abs(change - base) / (abs(base) + 1e-8)
    return (new_t.sub(old_t).abs()).div(old_t.abs().add(EPS))


@dataclass
class ParamSummaryRow:
    param_name: str
    layer_name: str
    average_diff: float
    numel: int


def _setup_logging(log_path: Optional[str] = None, verbose: bool = True) -> None:
    handlers = []
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    if verbose:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers if handlers else None,
    )


def _infer_device(device: str) -> torch.device:
    device = (device or "auto").lower()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device in ("cuda", "gpu"):
        return torch.device("cuda")
    if device == "cpu":
        return torch.device("cpu")
    # allow "cuda:0"
    return torch.device(device)


def _infer_dtype(dtype: str) -> torch.dtype:
    dtype = (dtype or "bf16").lower()
    if dtype in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype in ("fp16", "float16"):
        return torch.float16
    if dtype in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype}")


def _load_model(model_path: str, torch_dtype: torch.dtype) -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )


@torch.no_grad()
def _count_region_numel(model: AutoModelForCausalLM, region: str) -> Tuple[int, List[str]]:
    names = []
    total = 0
    for name, p in model.named_parameters():
        if _in_region(name, region):
            names.append(name)
            total += p.numel()
    return total, names


@torch.no_grad()
def _build_global_flat_diff(
    old_model: AutoModelForCausalLM,
    new_model: AutoModelForCausalLM,
    region: str,
    diff_device: torch.device,
) -> Tuple[torch.Tensor, List[ParamSummaryRow], List[Tuple[str, int, torch.Size]]]:
    """
    Build ONE flattened tensor containing the elementwise relative differences between 
    the new and old model parameters, restricted to the specified region.
    Returns:
      - flat_diff (same dtype as model params) on diff_device
      - matrix-level per-param summary (mean diff)
      - param_meta: list of (param_name, numel, shape) in the exact iteration order
    """
    # Ensure consistent order & names
    old_named = list(old_model.named_parameters())
    new_named = list(new_model.named_parameters())
    assert len(old_named) == len(new_named), "Parameter length mismatch between models."

    region_total, _ = _count_region_numel(new_model, region)
    if region_total == 0:
        raise ValueError(f"No parameters found for region={region}")

    # Keep dtype consistent with model parameters.
    param_dtype = next(new_model.parameters()).dtype
    flat_diff = torch.empty((region_total,), dtype=param_dtype, device=diff_device)

    summary_rows: List[ParamSummaryRow] = []
    param_meta: List[Tuple[str, int, torch.Size]] = []

    offset = 0
    for (n1, p_old), (n2, p_new) in zip(old_named, new_named):
        if n1 != n2:
            raise RuntimeError(f"Parameter mismatch: {n1} != {n2}")
        if not _in_region(n1, region):
            continue

        # Move tensors to diff_device only for diff computation, keep original dtype
        old_t = p_old.data.to(device=diff_device)
        new_t = p_new.data.to(device=diff_device)

        d = _relative_diff(old_t, new_t)
        numel = d.numel()

        flat_diff[offset : offset + numel].copy_(d.view(-1))

        # Summary uses .item() -> python float, independent of tensor dtype
        avg = float(d.mean().item())

        summary_rows.append(
            ParamSummaryRow(
                param_name=n1,
                layer_name=_layer_id(n1),
                average_diff=avg,
                numel=numel,
            )
        )
        param_meta.append((n1, numel, p_new.data.shape))

        offset += numel

        # free temp
        del old_t, new_t, d

    assert offset == region_total, f"Internal error: filled {offset} != allocated {region_total}"
    return flat_diff, summary_rows, param_meta


def _select_indices(
    flat_diff: torch.Tensor,
    k_percent: float,
    strategy: str,
    seed: int,
) -> Dict[str, object]:
    """
    Select k% parameters to restore based on the GLOBAL flat_diff.
    Returns a dict with:
      - selected_indices (1D LongTensor on flat_diff.device)
      - k_count
      - stats of selected values and overall values
    """
    strategy = (strategy or "top").lower()
    n = int(flat_diff.numel())

    k_count = int(n * float(k_percent) / 100.0)
    k_count = max(0, min(k_count, n))

    if k_count == 0:
        raise ValueError(f"k_count=0 (n={n}, k_percent={k_percent}). Please increase k_percent.")

    if strategy == "top":
        sel_vals, sel_idx = torch.topk(flat_diff, k_count)
    elif strategy == "bottom":
        sel_vals, sel_idx = torch.topk(-flat_diff, k_count)
        sel_vals = -sel_vals
    elif strategy == "random":
        rng = np.random.RandomState(seed)
        idx_np = rng.choice(n, k_count, replace=False)
        sel_idx = torch.tensor(idx_np, device=flat_diff.device, dtype=torch.long)
        sel_vals = flat_diff.index_select(0, sel_idx)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use top/bottom/random.")

    def _stats(x: torch.Tensor) -> Dict[str, float]:
        return {
            "max": float(x.max().item()),
            "median": float(x.median().item()),
            "mean": float(x.mean().item()),
            "min": float(x.min().item()),
            "sum": float(x.sum().item()),
        }

    return {
        "selected_indices": sel_idx,
        "k_count": k_count,
        "strategy": strategy,
        "selected_rel_diffs": _stats(sel_vals),
        "total_rel_diffs": _stats(flat_diff),
    }


@torch.no_grad()
def _restore_selected(
    old_model: AutoModelForCausalLM,
    new_model: AutoModelForCausalLM,
    param_meta: List[Tuple[str, int, torch.Size]],
    selected_indices: torch.Tensor,
    output_dir: str,
    device: torch.device,
) -> None:
    """
    Apply element-level restore: new_param[selected] = old_param[selected]
    Additionally, we record per-matrix restore counts and ratios into a TSV.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build a global boolean mask on the same device as selected_indices
    region_total = sum(numel for _, numel, _ in param_meta)
    global_mask = torch.zeros((region_total,), dtype=torch.bool, device=selected_indices.device)
    global_mask.scatter_(0, selected_indices, True)

    # Move models to target device for restore + save
    old_model.to(device)
    new_model.to(device)

    # Iterate params in the SAME order used to build param_meta
    old_sd = dict(old_model.named_parameters())
    new_sd = dict(new_model.named_parameters())

    restore_rows: List[Tuple[str, str, int, int, float]] = []

    offset = 0
    for name, numel, shape in param_meta:
        m = global_mask[offset : offset + numel].view(shape).to(device)
        p_new = new_sd[name]
        p_old = old_sd[name]

        # Element-level restore: selected positions take original value.
        # Equivalent to: p_new = (~mask)*p_new + mask*p_old
        p_new.data.copy_(torch.where(m, p_old.data, p_new.data))

        restored_numel = int(m.sum().item())
        restored_ratio = float(restored_numel / numel) if numel > 0 else 0.0
        restore_rows.append((name, _layer_id(name), restored_numel, int(numel), restored_ratio))

        offset += numel

    assert offset == region_total

    _save_restore_stats_tsv(restore_rows, os.path.join(output_dir, "restore_stats.tsv"))

    # Save restored model
    new_model.save_pretrained(output_dir)


def _save_matrix_summary_tsv(rows: List[ParamSummaryRow], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Keep it simple TSV: param_name, layer_name, average_diff, numel
    with open(path, "w", encoding="utf-8") as f:
        f.write("param_name\tlayer_name\taverage_mean_diff\tnumel\n")
        for r in rows:
            f.write(f"{r.param_name}\t{r.layer_name}\t{r.average_diff:.8f}\t{r.numel}\n")


def _save_selected_param_stats(payload: Dict[str, object], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _save_restore_stats_tsv(rows: List[Tuple[str, str, int, int, float]], path: str) -> None:
    """
    Save per-matrix restore counts:
      param_name, layer_name, restored_numel, numel, restored_ratio
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("param_name\tlayer_name\trestored_numel\tnumel\trestored_ratio\n")
        for param_name, layer_name, restored_numel, numel, restored_ratio in rows:
            f.write(f"{param_name}\t{layer_name}\t{restored_numel}\t{numel}\t{restored_ratio:.8f}\n")


def run(
    target_model_path: str,
    original_model_path: str,
    output_path: str,
    k_percent: float = 1.0,
    strategy: str = "top",  # top/bottom/random
    region: str = "transformer",  # transformer/embedding/mlp/attention
    device: str = "auto",  # auto/cpu/cuda/cuda:0...
    dtype: str = "bf16",   # bf16/fp16/fp32
    diff_device: str = "auto",  # where to compute flat diff + topk; usually same as device
    seed: int = 32,
    log_path: Optional[str] = None,
    verbose: bool = True,
):
    """
    Read original & target model, compute element-level relative diffs, select k% elements globally
    (top/bottom/random) inside region, restore those elements to original values, save output model.

    Outputs:
      - Restored model saved to output_path
      - matrix-level diff summary TSV: output_path/diff_summary.tsv
      - selected params stats JSON: output_path/selected_param_stats.json
      - per-matrix restore stats TSV: output_path/restore_stats.tsv
    """
    _setup_logging(log_path=log_path, verbose=verbose)

    device_t = _infer_device(device)
    diff_device_t = _infer_device(diff_device) if diff_device else device_t
    torch_dtype = _infer_dtype(dtype)

    logging.info(f"target_model_path={target_model_path}")
    logging.info(f"original_model_path={original_model_path}")
    logging.info(f"output_path={output_path}")
    logging.info(f"region={region}, strategy={strategy}, k_percent={k_percent}")
    logging.info(f"device={device_t}, diff_device={diff_device_t}, dtype={torch_dtype}")
    logging.info(f"seed={seed}")

    t0 = time.time()
    old_model = _load_model(original_model_path, torch_dtype=torch_dtype)
    new_model = _load_model(target_model_path, torch_dtype=torch_dtype)
    logging.info(f"Loaded models in {time.time() - t0:.2f}s")

    # Build global flat diff (same dtype as model parameters) for region parameters only
    t1 = time.time()
    flat_diff, summary_rows, param_meta = _build_global_flat_diff(
        old_model=old_model,
        new_model=new_model,
        region=region,
        diff_device=diff_device_t,
    )
    logging.info(f"Built flat diff: numel={flat_diff.numel()} in {time.time() - t1:.2f}s")

    # Select indices (top/bottom/random) on flat_diff
    t2 = time.time()
    selection_info = _select_indices(flat_diff=flat_diff, k_percent=k_percent, strategy=strategy, seed=seed)
    sel_idx: torch.Tensor = selection_info["selected_indices"]
    logging.info(f"Selected k_count={selection_info['k_count']} in {time.time() - t2:.2f}s")
    logging.info(f"Selected stats: {selection_info['selected_stats']}")
    logging.info(f"Total stats: {selection_info['total_stats']}")

    # Save summaries (before restore, they describe original diff between models)
    _save_matrix_summary_tsv(summary_rows, os.path.join(output_path, "diff_summary.tsv"))
    _save_selected_param_stats(selection_info, os.path.join(output_path, "selected_param_stats.json"))

    # Restore selected elements (also writes restore_stats.tsv)
    t3 = time.time()
    _restore_selected(
        old_model=old_model,
        new_model=new_model,
        param_meta=param_meta,
        selected_indices=sel_idx,
        output_dir=output_path,
        device=device_t,
    )
    logging.info(f"Restored + saved model in {time.time() - t3:.2f}s")

    # Cleanup
    del flat_diff, sel_idx
    del old_model, new_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logging.info("Done.")


def main():
    fire.Fire({"run": run})


if __name__ == "__main__":
    main()
