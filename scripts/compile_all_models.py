"""
compile_all_models.py
 
Orchestrator only — loops over every .onnx file in models/incoming/ and
calls compile_model.py once per file as a subprocess. Contains no
inspection or compile logic itself, so compile_model.py is always the
single source of truth for how one model gets compiled.
 
Usage:
    python3 scripts/compile_all_models.py
    python3 scripts/compile_all_models.py --clear-stale
"""
 
import argparse
import glob
import json
import os
import subprocess
import sys
 
 
def main():
    parser = argparse.ArgumentParser(description="Compile every ONNX model in incoming-dir, one by one.")
    parser.add_argument("--incoming-dir", default="models/quantized")
    parser.add_argument("--output-dir", default="models/engines")
    parser.add_argument("--clear-stale", action="store_true",
                         help="Delete existing .engine files in output-dir before compiling. "
                              "Recommended after any TensorRT/container version change.")
    args = parser.parse_args()
 
    os.makedirs(args.output_dir, exist_ok=True)
 
    if args.clear_stale:
        stale = glob.glob(os.path.join(args.output_dir, "*.engine"))
        for f in stale:
            os.remove(f)
        if stale:
            print(f"🧹 Removed {len(stale)} stale engine(s)")
 
    onnx_models = sorted(glob.glob(os.path.join(args.incoming_dir, "*.onnx")))
    if not onnx_models:
        print(f"⚠️  No ONNX models found in {args.incoming_dir}/")
        return
 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    compile_script = os.path.join(script_dir, "compile_model.py")
 
    results = []
    for model_path in onnx_models:
        filename = os.path.basename(model_path)
        print(f"\n{'=' * 60}")
        print(f"  {filename}")
        print(f"{'=' * 60}")
 
        cmd = [
            sys.executable, compile_script,
            "--model", model_path,
            "--output-dir", args.output_dir,
        ]
        result = subprocess.run(cmd)
 
        if result.returncode == 0:
            results.append({"file": filename, "status": "success"})
        elif result.returncode == 2:
            results.append({"file": filename, "status": "skipped"})
        else:
            results.append({"file": filename, "status": "failed"})
 
    manifest_path = os.path.join(args.output_dir, "compile_all_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"results": results}, f, indent=2)
 
    succeeded = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = sum(1 for r in results if r["status"] == "failed")
    print(f"\n{'=' * 60}")
    print(f"📋 {succeeded} compiled, {skipped} skipped, {failed} failed.")
    print(f"📄 {manifest_path}")
 
 
if __name__ == "__main__":
    main()
