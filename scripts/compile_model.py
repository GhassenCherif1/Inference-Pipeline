"""
compile_model.py

Compiles ONE ONNX model into a TensorRT engine on Jetson Orin Nano.

Calls inspect_model.py first to find out what's really inside the
ONNX graph (rather than trusting the filename), then builds the
correct trtexec command — or refuses to compile if the precision
can't be correctly run on this hardware (e.g. true FP8).

Usage:
    python3 scripts/compile_model.py --model models/incoming/yolo_int8.onnx
    python3 scripts/compile_model.py --model models/incoming/yolo_int8.onnx --workspace-mb 1024
"""

import argparse
import os
import subprocess
import sys

# Reuse the inspection logic directly rather than re-implementing it here.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inspect_model import inspect_onnx, summarize


def get_trt_arch_tag() -> str:
    """Tag engine filenames with TRT version + GPU arch so stale engines
    from a previous container build are never silently reused."""
    trt_version = "unknown"
    try:
        result = subprocess.run(["trtexec", "--version"], capture_output=True, text=True, timeout=10)
        for line in (result.stdout + result.stderr).splitlines():
            if "TensorRT" in line and "version" in line.lower():
                trt_version = line.strip().split()[-1]
                break
    except Exception:
        pass

    sm_tag = "sm-unknown"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        cap = result.stdout.strip().replace(".", "")
        if cap:
            sm_tag = f"sm{cap}"
    except Exception:
        pass

    return f"trt{trt_version}-{sm_tag}"


def build_trtexec_command(info: dict, engine_path: str):
    """Returns the trtexec command, or None if this model should be skipped."""
    cmd = [
        "trtexec",
        f"--onnx={info['filepath']}",
        f"--saveEngine={engine_path}",
	"--profilingVerbosity=detailed",
	"--useCudaGraph",
	"--useSpinWait",
	"--builderOptimizationLevel=5"
    ]

    filename_lower = info["filename"].lower()
    weight_dtypes = set(info.get("weight_dtypes", []))
    has_qdq = info.get("has_qdq_nodes", False)
    has_int4 = info.get("has_int4_pack", False)

    if "fp8" in filename_lower or "float8" in weight_dtypes:
        print(f"   ⛔ FP8 detected. Orin Nano (Ampere) has no native FP8 tensor cores — "
              f"this needs Ada/Hopper/Blackwell. Refusing to silently recompile as FP16.")
        return None

    if "int4" in filename_lower or has_int4:
        if not has_int4 and not has_qdq:
            print(f"   ⛔ Filename suggests INT4 but no INT4 packing or QDQ nodes were found "
                  f"({info['summary']}). Likely a dense export — refusing to compile as INT4.")
            return None
        print(f"   ⚠️  INT4 weight-only patterns detected. trtexec has no generic --int4 flag; "
              f"compiling with --fp16 compute. VERIFY actual engine precision after — "
              f"don't assume the label is correct.")
        cmd.append("--fp16")
        return cmd

    if "int8" in filename_lower:
        if not has_qdq:
            print(f"   ⛔ Filename suggests INT8 but no QuantizeLinear/DequantizeLinear nodes "
                  f"were found ({info['summary']}). Refusing to compile — re-check the ModelOpt export.")
            return None
        cmd.extend(["--int8", "--fp16"])
        return cmd

    if "fp16" in filename_lower or "float16" in weight_dtypes:
        cmd.append("--fp16")
        return cmd

    print(f"   ℹ️  No quantization markers ({info['summary']}). Compiling dense FP32 baseline.")
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Compile a single ONNX model to a TensorRT engine.")
    parser.add_argument("--model", required=True, help="Path to the .onnx file to compile")
    parser.add_argument("--output-dir", default="models/engines")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        raise SystemExit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"🔍 Inspecting {os.path.basename(args.model)}")
    info = inspect_onnx(args.model)
    info["summary"] = summarize(info)
    print(f"   {info['summary']}")
    for note in info["notes"]:
        print(f"   📝 {note}")

    if info["parse_error"]:
        print(f"❌ Cannot compile — ONNX graph failed to parse ({info['parse_error']})")
        raise SystemExit(1)

    arch_tag = get_trt_arch_tag()
    base_name = os.path.splitext(info["filename"])[0]
    engine_path = os.path.join(args.output_dir, f"{base_name}__{arch_tag}.engine")

    print(f"🛠️  Compiling (build tag: {arch_tag})")
    cmd = build_trtexec_command(info, engine_path)
    if cmd is None:
        print("⏭️  Skipped — precision unsupported or mismatched. See messages above.")
        raise SystemExit(2)

    print(f"   Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Compiled: {engine_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Compilation failed")
        tail = e.stderr.strip().splitlines()[-15:] if e.stderr else []
        print("   " + "\n   ".join(tail))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
