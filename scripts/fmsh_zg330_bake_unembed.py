#!/usr/bin/env python3
"""Offline bake tool: extract a single named GGUF tensor (the LM-head / tied
unembedding weight) and compile it as a *constant* matmul network for the
FMSH ZG330 backend.

Runs once per (model, quant, m, qdtype) on an x86 dev host inside the
`fpai-icraft:latest` docker image (ARM cannot run `icraft compile`). Produces
exactly one network for exactly one fixed `M` — no bucket list, no multiple
shapes. The ARM/x64 runtime (`ggml-fmsh-zg330.cpp`) only ever *loads* this
artifact; it never compiles.

Usage:
    python3 scripts/fmsh_zg330_bake_unembed.py \\
        --gguf Qwen3.5-0.8B-Q4_K_M.gguf \\
        --tensor-name token_embd.weight \\
        --m 1 \\
        --qdtype bf16 \\
        --cache-dir .cache/deploy \\
        --net-name unembed
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if "NO_LOCAL_GGUF" not in os.environ and (REPO_ROOT / "gguf-py").exists():
    sys.path.insert(0, str(REPO_ROOT / "gguf-py"))

import gguf  # noqa: E402
from gguf.gguf_reader import GGUFReader  # noqa: E402

try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
except ImportError as e:
    raise SystemExit(
        "missing python package 'onnx' — run this script inside the fpai-icraft docker image"
    ) from e


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gguf", required=True, type=Path, help="path to the source GGUF model file")
    p.add_argument(
        "--tensor-name",
        default="token_embd.weight",
        help="GGUF tensor name to bake (default: token_embd.weight, the tied LM-head weight)",
    )
    p.add_argument("--m", type=int, default=1, help="fixed row count M for the baked matmul (default: 1)")
    p.add_argument(
        "--qdtype",
        choices=["bf16", "int8"],
        default="bf16",
        help="on-device weight precision. fp32/tf32 are intentionally not offered for this weight "
             "(requirement B: f32 forbidden). bf16 needs no calibration; int8 is opt-in and needs "
             "--calib-dir/--calib-list",
    )
    p.add_argument("--calib-dir", type=Path, default=None, help="calibration samples dir, required if --qdtype int8")
    p.add_argument("--calib-list", type=Path, default=None, help="calibration file list, required if --qdtype int8")
    p.add_argument("--cache-dir", type=Path, default=Path(".cache/deploy"), help="cache root the runtime will read from")
    p.add_argument("--net-name", default="unembed", help="network name prefix (default: unembed)")
    p.add_argument(
        "--external-data",
        action="store_true",
        help="force ONNX external-data storage for the initializer instead of trying inline raw_data first",
    )
    p.add_argument("--skip-compile", action="store_true", help="write ONNX + toml but skip running `icraft compile`")
    return p.parse_args()


def find_tensor(reader: GGUFReader, name: str):
    for t in reader.tensors:
        if t.name == name:
            return t
    available = ", ".join(sorted(t.name for t in reader.tensors if "embd" in t.name or "output" in t.name))
    raise SystemExit(f"tensor '{name}' not found in GGUF. candidates: {available}")


def build_const_matmul_onnx(onnx_path: Path, weight_kn: np.ndarray, m: int, external_data: bool) -> None:
    """weight_kn: float32 ndarray, shape [k, n] (netmake's B[k,n] convention)."""
    k, n = weight_kn.shape
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [m, k])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [m, n])
    # ONNX-parse-stage intermediate representation only; the actual on-device
    # stored precision is controlled by icraft's [quantize] qdtype, not this dtype.
    W = numpy_helper.from_array(np.ascontiguousarray(weight_kn, dtype=np.float32), name="W")

    node = helper.make_node("MatMul", inputs=["A", "W"], outputs=["Y"], name="MatMul_0")
    graph = helper.make_graph([node], "unembed_const_matmul_graph", [A], [Y], initializer=[W])
    model = helper.make_model(graph, producer_name="ggml_fmsh_bake_unembed", opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 8

    def _save(as_external: bool) -> None:
        if as_external:
            data_file = onnx_path.name + ".data"
            data_path = onnx_path.parent / data_file
            if data_path.exists():
                data_path.unlink()
            onnx.save_model(
                model,
                str(onnx_path),
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=data_file,
                size_threshold=1024,
            )
        else:
            onnx.save(model, str(onnx_path))

    if external_data:
        print(f"[bake] saving ONNX with external data (forced): {onnx_path}")
        _save(True)
        return

    try:
        print(f"[bake] attempting inline (raw_data) ONNX save: {onnx_path} (~{weight_kn.nbytes / 1e6:.1f} MB weight)")
        _save(False)
        onnx.checker.check_model(str(onnx_path))
        print("[bake] inline ONNX save + checker OK")
    except Exception as e:  # noqa: BLE001 - documented empirical fallback, not an assumed failure mode
        print(f"[bake] inline ONNX save/check failed ({e!r}); falling back to external data")
        _save(True)


def write_icraft_compile_toml(
    work_dir: Path,
    net_name: str,
    onnx_name: str,
    m: int,
    k: int,
    qdtype: str,
    calib_dir: Path | None,
    calib_list: Path | None,
) -> Path:
    toml_path = work_dir / f"{net_name}.toml"
    jr = f"./.cache/{net_name}_ZG/"

    lines = [
        "[parse]",
        f'net_name = "{net_name}"',
        'framework = "Onnx"',
        f'network = "{onnx_name}"',
        f'jr_path = "{jr}"',
        'target = "zhuge"',
        f"inputs = [[{m},{k}]]",
        'inputs_layout = "FD"',
        'inputs_dtype = "fp32"',
        'pre_method = "nop"',
        'pre_mean = "nop"',
        'pre_scale = "nop"',
        'channel_swap = "nop"',
        "",
        "[optimize]",
        f'json = "{jr}{net_name}_parsed.json"',
        f'raw = "{jr}{net_name}_parsed.raw"',
        f'jr_path = "{jr}"',
        "",
        "[quantize]",
        f'json = "{jr}{net_name}_optimized.json"',
        f'raw = "{jr}{net_name}_optimized.raw"',
        f'jr_path = "{jr}"',
        'target = "zhuge"',
        f'qdtype = "{qdtype}"',
        'forward_mode = "image"',
    ]
    if qdtype == "int8":
        if not calib_dir or not calib_list:
            raise SystemExit("--qdtype int8 requires --calib-dir and --calib-list (see docs/quantizer/zhuge/zhuge.md)")
        lines += [
            f'forward_dir = "{calib_dir}"',
            f'forward_list = "{calib_list}"',
        ]
    lines += [
        "",
        "[adapt]",
        f'json = "{jr}{net_name}_quantized.json"',
        f'raw = "{jr}{net_name}_quantized.raw"',
        f'jr_path = "{jr}"',
        "",
        "[generate]",
        f'json = "{jr}{net_name}_adapted.json"',
        f'raw = "{jr}{net_name}_adapted.raw"',
        f'jr_path = "{jr}"',
        "",
    ]

    toml_path.write_text("\n".join(lines))
    return toml_path


def run_icraft_compile(work_dir: Path, toml_path: Path) -> None:
    cmd = ["icraft", "compile", toml_path.name]
    print(f"[bake] running: (cd {work_dir} && {' '.join(cmd)})")
    subprocess.run(cmd, cwd=str(work_dir), check=True)


def find_generated_zg_json_raw(work_dir: Path, net_name: str) -> tuple[Path, Path]:
    cache_root = work_dir / ".cache"
    cand_json = cache_root / f"{net_name}_ZG.json"
    cand_raw = cache_root / f"{net_name}_ZG.raw"
    if cand_json.exists() and cand_raw.exists():
        return cand_json, cand_raw
    if cache_root.exists():
        found_json = found_raw = None
        for p in cache_root.rglob(f"{net_name}_ZG.json"):
            found_json = p
        for p in cache_root.rglob(f"{net_name}_ZG.raw"):
            found_raw = p
        if found_json and found_raw:
            return found_json, found_raw
    raise SystemExit(f"cannot find generated *_ZG.json/raw under: {cache_root}")


def main() -> None:
    args = parse_args()
    if args.m <= 0:
        raise SystemExit("--m must be > 0")

    t0 = time.time()
    reader = GGUFReader(str(args.gguf))
    tensor = find_tensor(reader, args.tensor_name)
    print(f"[bake] tensor={tensor.name} type={tensor.tensor_type.name} shape(ne)={tuple(tensor.shape.tolist())}")

    # ReaderTensor.data is the raw quant-block bytes, numpy shape (n_vocab, n_embd)
    # after gguf's byte-shape conversion; dequantize() turns it into float32 of the
    # same logical shape (n_vocab, n_embd).
    dequantized = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
    dequantized = np.asarray(dequantized, dtype=np.float32)
    if dequantized.ndim != 2:
        raise SystemExit(f"expected a 2D tensor, got shape {dequantized.shape}")
    n_vocab, n_embd = dequantized.shape
    print(f"[bake] dequantized shape=(n_vocab={n_vocab}, n_embd={n_embd})")

    # netmake's B[k,n] convention: transpose once, offline, instead of every token
    # (this is the exact transform ggml-fmsh-zg330.cpp:1251-1259 does per-call).
    weight_kn = np.ascontiguousarray(dequantized.T, dtype=np.float32)  # [k=n_embd, n=n_vocab]
    k, n = weight_kn.shape
    m = args.m

    net_name = f"{args.net_name}_{m}x{k}x{n}"
    work_dir = args.cache_dir / net_name
    work_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = work_dir / f"{net_name}.onnx"

    build_const_matmul_onnx(onnx_path, weight_kn, m, args.external_data)

    toml_path = write_icraft_compile_toml(
        work_dir, net_name, onnx_path.name, m, k, args.qdtype, args.calib_dir, args.calib_list
    )
    print(f"[bake] wrote toml: {toml_path}")

    if args.skip_compile:
        print("[bake] --skip-compile set; not running icraft compile")
        return

    run_icraft_compile(work_dir, toml_path)
    json_path, raw_path = find_generated_zg_json_raw(work_dir, net_name)

    elapsed = time.time() - t0
    print("[bake] manifest:")
    print(f"  net_name   = {net_name}")
    print(f"  shape      = m={m} k={k} n={n}")
    print(f"  qdtype     = {args.qdtype}")
    print(f"  json       = {json_path} ({json_path.stat().st_size} bytes)")
    print(f"  raw        = {raw_path} ({raw_path.stat().st_size} bytes)")
    print(f"  cache_dir  = {args.cache_dir}")
    print(f"  wall_time  = {elapsed:.1f}s")


if __name__ == "__main__":
    main()
