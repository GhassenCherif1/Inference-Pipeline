"""
inspect_model.py

Inspects ONE ONNX model's actual graph to determine what quantization
is really present (Q/DQ nodes, weight dtypes, INT4 packing markers)
rather than trusting the filename. Prints the result and optionally
writes it to a JSON file for compile_model.py to consume.

Usage:
    python3 scripts/inspect_model.py --model models/incoming/yolo_int8.onnx
"""

import argparse
import json
import os

try:
    import onnx
except ImportError:
    print("❌ The 'onnx' package is required. Install with: pip install onnx --break-system-packages")
    raise SystemExit(1)


_DTYPE_NAMES = {
    1: "float32",
    10: "float16",
    3: "int8",
    6: "int32",
    2: "uint8",
}


def inspect_onnx(model_path: str) -> dict:
    """Returns a dict describing what's actually inside the ONNX graph."""
    info = {
        "filename": os.path.basename(model_path),
        "filepath": model_path,
        "has_qdq_nodes": False,
        "qdq_dtypes": [],
        "weight_dtypes": [],
        "has_int4_pack": False,
        "notes": [],
        "parse_error": "",
    }

    try:
        model = onnx.load(model_path, load_external_data=False)
    except Exception as e:
        info["parse_error"] = str(e)
        return info

    qdq_ops = {"QuantizeLinear", "DequantizeLinear"}
    qdq_dtypes, weight_dtypes = set(), set()

    for node in model.graph.node:
        if node.op_type in qdq_ops:
            info["has_qdq_nodes"] = True
        if "int4" in node.op_type.lower() or "woq" in node.op_type.lower():
            info["has_int4_pack"] = True
        for attr in node.attribute:
            if attr.name.lower() in ("bits", "quant_bits") and attr.i == 4:
                info["has_int4_pack"] = True

    for initializer in model.graph.initializer:
        dtype_name = _DTYPE_NAMES.get(initializer.data_type)
        if dtype_name:
            weight_dtypes.add(dtype_name)
        if initializer.data_type == 3:  # INT8
            qdq_dtypes.add("int8")

    info["qdq_dtypes"] = sorted(qdq_dtypes)
    info["weight_dtypes"] = sorted(weight_dtypes)

    if "modelopt" in (model.producer_name or "").lower():
        opset = model.opset_import[0].version if model.opset_import else "?"
        info["notes"].append(f"Exported by {model.producer_name} (opset {opset})")

    return info


def summarize(info: dict) -> str:
    if info["parse_error"]:
        return f"could not parse graph ({info['parse_error']})"
    parts = []
    if info["has_qdq_nodes"]:
        parts.append(f"QDQ nodes present (dtypes: {info['qdq_dtypes']})")
    if info["has_int4_pack"]:
        parts.append("INT4 weight-packing markers present")
    if info["weight_dtypes"]:
        parts.append(f"weight dtypes: {info['weight_dtypes']}")
    return "; ".join(parts) if parts else "no explicit quantization markers found (dense FP32/FP16)"


def main():
    parser = argparse.ArgumentParser(description="Inspect a single ONNX model's real quantization content.")
    parser.add_argument("--model", required=True, help="Path to the .onnx file to inspect")
    parser.add_argument("--out", default=None,
                         help="Optional path to write the inspection result as JSON "
                              "(used by compile_model.py if provided)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        raise SystemExit(1)

    print(f"🔍 {os.path.basename(args.model)}")
    info = inspect_onnx(args.model)
    info["summary"] = summarize(info)

    print(f"   {info['summary']}")
    for note in info["notes"]:
        print(f"   📝 {note}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(info, f, indent=2)
        print(f"📄 Written: {args.out}")

    return info


if __name__ == "__main__":
    main()
