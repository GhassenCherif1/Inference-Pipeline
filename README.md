# ViT Quantization Benchmarking — Jetson Container

This is the **second half** of a two-container compression, evaluation, and
benchmarking framework for Vision Transformers (with DINOv3 as the use case)
applied to object detection and segmentation.

- **Container 1 (server, not in this repo):** quantization and calibration
  with NVIDIA Model Optimizer, requiring a powerful GPU/VRAM. Produces
  flattened and quantized ONNX models.
- **Container 2 (this repo, Jetson Orin Nano):** takes those ONNX models,
  compiles them into TensorRT engines, runs inference, and benchmarks
  accuracy / latency / power / memory on real edge hardware.

The point of this container is to answer one question honestly: **does a
given quantized model actually run correctly and efficiently on a Jetson,
and what does it cost in accuracy to get there?**

## Why this exists / design philosophy

Quantized ONNX exports are easy to mislabel or silently misinterpret —
filenames say `int4` or `int8` but the graph doesn't always back that up,
and not every precision is even valid on Orin Nano's Ampere tensor cores
(e.g. FP8). So this container never trusts a filename:

1. Every model is **inspected** (`inspect_model.py`) before compilation to
   determine what's actually inside the ONNX graph (QDQ nodes, weight
   dtypes, INT4 packing markers).
2. Compilation (`compile_model.py`) uses that inspection result to pick the
   correct `trtexec` flags, or **refuses to compile** if the precision is
   unsupported on this hardware or doesn't match what the graph actually
   contains.
3. Engines are filename-stamped with the TensorRT version and GPU compute
   capability (`trt{version}-sm{cap}`) so a stale engine from a previous
   container build is never silently reused after a TensorRT/driver
   upgrade.
4. Benchmarking always measures the **real, deployed artifact** (the
   compiled `.engine`), not the ONNX model, using Jetson-native tooling
   (`tegrastats`) since standard NVML calls don't apply to Jetson's unified
   memory architecture.

## Repository structure

```
.
├── docker-compose.yml          # Service definition (nvidia runtime, host network/ipc)
├── Dockerfile.jetson           # l4t-tensorrt base image + Python deps
├── scripts/
│   ├── inspect_model.py                  # Ground-truth ONNX graph inspection
│   ├── compile_model.py                  # Inspect + compile ONE model -> TensorRT engine
│   ├── compile_all_models.py             # Orchestrator: compiles every .onnx in models/quantized/
│   ├── test_inference_with_tensorrt_engine.py   # Single-image smoke test for an engine
│   ├── benchmark_coco_object_detection.py       # Full COCO val2017 benchmark (latency/mAP/CO2/RAM/GPU)
│   ├── benchmark_batch_coco_object_detection.py        # Same, but with configurable batch size
│   ├── benchmark_batch_custom_object_detection.py     # Same, but against any custom COCO-format dataset
│   └── benchmark_segmentation.py                # mIoU benchmark for segmentation engines (e.g. DINOv3-EoMT)
├── models/
│   ├── onnx_models/
│   │   ├── flattened/          # Flattened ONNX models received from the quantization container
│   │   └── quantized/          # Quantized ONNX models received from the quantization container
│   ├── engines/                # Compiled TensorRT engines (+ layer_info / profile JSON)
│   ├── profiling/               # Per-precision layer_info + profile JSON from trtexec
│   ├── jetson_telemetry.csv     # Output of detection benchmark runs
│   ├── segmentation_telemetry.csv # Output of segmentation benchmark runs
│   └── carbon_raw.csv           # Raw CodeCarbon output
├── data/
│   ├── coco_dataset/            # COCO val2017 images + annotations
│   └── cocostuff/                # COCO-Stuff images + stuff masks (for segmentation)
└── ...
```

## Container setup

Built on top of NVIDIA's `l4t-tensorrt:r10.3.0-devel` image, with Python
tooling for benchmarking layered on top (OpenCV, psutil, CodeCarbon,
pycocotools, onnx, nvidia-ml-py).

```bash
docker compose build
docker compose run benchmarker
```

`docker-compose.yml` mounts the repo into `/workspace`, bind-mounts the
host's `tegrastats` binary read-only (it needs to run against the host's
Jetson hardware monitoring, not a containerized one), and grants the
`nvidia` runtime with `compute,utility` capabilities.

## Workflow

### 1. Inspect a model (optional standalone step)

```bash
python3 scripts/inspect_model.py --model models/onnx_models/quantized/convnext-base-ltdetr-coco-flat-int8-quantized.onnx
```

Reports whether QDQ nodes, INT8 weights, or INT4 packing markers are
actually present in the graph — independent of what the filename claims.

### 2. Compile

Single model:

```bash
python3 scripts/compile_model.py --model models/onnx_models/quantized/<model>.onnx
```

All models in `models/onnx_models/quantized/`:

```bash
python3 scripts/compile_all_models.py --clear-stale
```

Compilation logic per detected precision:

| Detected in graph | Behavior |
|---|---|
| `fp8` in filename / float8 weights | **Refused** — Orin Nano (Ampere) has no native FP8 tensor cores |
| `int4` in filename, with INT4/QDQ markers present | Compiles with `--fp16` (trtexec has no generic INT4 flag); engine precision must be verified post-hoc |
| `int4` in filename, no markers found | **Refused** — likely a dense export mislabeled as INT4 |
| `int8` in filename, QDQ nodes present | Compiles with `--int8 --fp16` |
| `int8` in filename, no QDQ nodes | **Refused** — filename doesn't match graph contents |
| `fp16` in filename / float16 weights | Compiles with `--fp16` |
| No markers found | Compiles as dense FP32 baseline |

Every engine is built with `--profilingVerbosity=detailed --useCudaGraph
--useSpinWait --builderOptimizationLevel=5` and saved as
`{model_name}__trt{version}-sm{compute_cap}.engine` under
`models/engines/`. `compile_all_models.py` writes a
`compile_all_manifest.json` summarizing success/skip/fail per model.

### 3. Smoke-test a single engine

```bash
python3 scripts/test_inference_with_tensorrt_engine.py
```

Runs one real image through a hardcoded engine path, times pure GPU
execution with CUDA events, and writes an annotated output image to
`data/output_with_boxes.jpg`. Useful as a fast sanity check before running
a full benchmark sweep.

### 4. Benchmark

All four benchmark scripts share the same engine-loading, CUDA-event
timing, `tegrastats`-based GPU utilization sampling, psutil-based unified
RAM delta tracking, and CodeCarbon emissions tracking, so their CSV output
is directly comparable.

**COCO object detection (single image at a time):**

```bash
python3 scripts/benchmark_coco_object_detection.py \
  --engine models/engines/<engine_file> \
  --annotations data/coco_dataset/annotations/instances_val2017.json \
  --image_dir data/coco_dataset/images/val2017/ \
  --metrics_csv models/jetson_telemetry.csv
```

Reports latency, throughput, peak unified-RAM delta, average/peak GPU
utilization, CO2, mAP@[.5:.95], mAP@.5, detection count, and skipped
images, appended as one row per engine to the metrics CSV.

**COCO object detection (batched):**

```bash
python3 scripts/benchmark_batch_coco_object_detection.py \
  --engine models/engines/<engine_file> \
  --annotations data/coco_dataset/annotations/instances_val2017.json \
  --image_dir data/coco_dataset/images/val2017/ \
  --metrics_csv models/jetson_telemetry.csv \
  --batch_size 8
```

Same as the single-image script, but `--batch_size` is now a CLI argument instead of single-image inference — use it to characterize throughput at real deployment batch sizes. `--batch_size 1` matches the single-image script exactly (with added Batch_Size/Avg_Batch_Latency_ms columns) for direct comparison across batch sizes.

**Custom dataset object detection:**

```bash
python3 scripts/benchmark_batch_custom_object_detection.py \
  --engine models/engines/<engine_file> \
  --image_dir <your_images> \
  --annotations <your_coco_format_json> \  # optional — omit to skip mAP
  --batch_size 8
```

Accepts any folder of images and an optional COCO-format annotations file
(exportable from Roboflow, CVAT, Label Studio, FiftyOne, etc.) — no
dependency on COCO's specific 80-class taxonomy. An optional
`--category_map` JSON remaps model output indices to your dataset's
category ids when they don't already match. Supports the same `--batch_size`
argument as the batched COCO script (default 1, reproducing single-image
inference), with the same `Batch_Size`/`Avg_Batch_Latency_ms` columns
appended to `--metrics_csv` for comparing batch sizes on the same engine.

**Segmentation:**

```bash
python3 scripts/benchmark_segmentation.py \
  --engine models/engines/<segmentation_engine> \
  --category_map seg_category_map.json
```

Benchmarks mIoU against COCO-Stuff-style single-channel indexed PNG ground
truth masks. `--category_map` translates the model's internal class index
space to the dataset's real category ids (required for lightly-train
DINOv3-EoMT checkpoints, whose internal indices have gaps relative to
COCO-Stuff's 1–182 ids — see the script's docstring for how to regenerate
this map from `model.internal_class_to_class`).

## Metrics collected

Every benchmark run records, per engine:

- **Latency** (ms) and **throughput** (FPS), timed with CUDA events around
  the actual GPU execution (upload → inference → download), not wall clock
- **Peak unified RAM delta** (MB), via psutil — Orin Nano has no separate
  VRAM pool, so system RAM is the right signal
- **Avg / peak GPU utilization** (%), sampled from `tegrastats` on a
  background thread (`GR3D_FREQ`); left blank automatically when not
  running on real Jetson hardware
- **CO2 emissions** (kg), via CodeCarbon (offline tracker, Germany grid
  factor)
- **Accuracy**: mAP@[.5:.95] / mAP@.5 (pycocotools) for detection, mIoU for
  segmentation
- **Images skipped** (missing files, unreadable images, per-image
  processing errors — a run never hard-crashes on one bad image)

Results accumulate as rows in `models/jetson_telemetry.csv` /
`models/segmentation_telemetry.csv`, so every compiled precision variant of
a model (FP32, FP16, INT8 with/without ModelOpt, INT4) can be compared
side by side: accuracy lost vs. speed/memory/power gained.

## A note on trust

This framework is built around **verifying claims about precision rather
than assuming them** — at inspection time (graph contents vs. filename), at
compile time (refusing mismatched/unsupported precisions instead of
silently downgrading), and the INT4 path explicitly says don't assume the
label is correct, since trtexec compiles it as FP16 compute under the
hood. Treat engine filenames as a hypothesis, and the `profiling/` and
`layer_info` JSON outputs (post-compile) as the way to confirm it.