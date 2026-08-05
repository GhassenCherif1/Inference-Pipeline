#!/usr/bin/env python3
# scripts/benchmark_custom.py
"""
Benchmarks one compiled TensorRT engine against a CUSTOM dataset, with
configurable batch size: latency, throughput, peak RAM delta, GPU
utilization, CO2 (CodeCarbon), and mAP (optional).

This is a sibling to benchmark_coco_object_detection.py /
benchmark_batch_coco_detection.py. All the engine-loading, batched
inference loop, and telemetry code (tegrastats, CodeCarbon, RAM tracking)
is the same. The only thing that changes is how ground truth and category
mappings are read in, so this can work with ANY dataset instead of only
COCO val2017.

----------------------------------------------------------------------
HOW TO BRING YOUR OWN DATASET
----------------------------------------------------------------------
1. Images: any folder of images. Pass it via --image_dir.

2. Ground truth (optional — omit --annotations entirely to skip mAP and
   only get latency/throughput/CO2/GPU telemetry):
   Must be a COCO-FORMAT json file, i.e. a dict with "images",
   "annotations", and "categories" keys, like:
     {
       "images": [{"id": 1, "file_name": "img001.jpg", ...}, ...],
       "annotations": [{"image_id": 1, "category_id": 0,
                         "bbox": [x, y, w, h], ...}, ...],
       "categories": [{"id": 0, "name": "my_class"}, ...]
     }
   Many labeling tools (Roboflow, CVAT, Label Studio, FiftyOne) can
   export directly to this format — look for an "Export as COCO" /
   "COCO JSON" option. This script does NOT require COCO's specific
   80-class taxonomy or its non-contiguous category ids; your
   categories can be anything, numbered however you like.

3. Category mapping (--category_map, optional):
   Your model's output class indices (0, 1, 2, ...) need to map to the
   category_id values used in your annotations json. If your model was
   trained with class index i corresponding directly to category_id i
   (the common case for a freshly-trained custom model), you don't need
   to pass anything — the default is an identity mapping.
   If your indices and category_ids differ, pass a JSON file:
     {"0": 5, "1": 12, "2": 1}
   meaning "model output index 0 -> category_id 5", etc.

----------------------------------------------------------------------
BATCH SIZE
----------------------------------------------------------------------
--batch_size (default 1) controls how many images are pushed through the
engine per inference call. Set it to characterize throughput at the batch
sizes your deployment will actually use; --batch_size 1 reproduces plain
single-image inference. Output rows carry a Batch_Size column so runs at
different batch sizes for the same engine can be compared directly.

----------------------------------------------------------------------
WHAT'S THE SAME AS benchmark_coco_object_detection.py / benchmark_batch_coco_detection.py
----------------------------------------------------------------------
  - TensorRT engine loading, I/O tensor discovery, batched inference loop
  - tegrastats-based GPU utilization sampling (Jetson-specific; see
    benchmark_coco_object_detection.py's docstring for why pynvml isn't used)
  - psutil-based unified RAM delta tracking
  - CodeCarbon emissions tracking
  - Output CSV schema (same columns, so results from all three scripts can
    be appended to the same metrics file and compared directly)
"""
import argparse
import os
import csv
import json
import re
import shutil
import subprocess
import threading
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import psutil
from codecarbon import OfflineEmissionsTracker

parser = argparse.ArgumentParser()
parser.add_argument("--engine", type=str, required=True, help="Path to compiled stamped .engine binary")
parser.add_argument("--image_dir", type=str, required=True, help="Folder of images to run inference on")
parser.add_argument("--annotations", type=str, default=None,
                     help="Optional path to a COCO-FORMAT json ground-truth file. "
                          "If omitted, mAP is skipped and only speed/telemetry metrics are recorded.")
parser.add_argument("--category_map", type=str, default=None,
                     help="Optional path to a JSON file mapping model output index (string key) "
                          "-> annotation category_id (int value), e.g. {\"0\": 5, \"1\": 12}. "
                          "If omitted, an identity mapping is assumed (index i -> category_id i).")
parser.add_argument("--metrics_csv", type=str, default="results/custom_detection_batch.csv")
parser.add_argument("--input_shape", type=str, default="1,3,640,640",
                     help="Comma-separated NCHW shape. The N here is ignored — N is set from "
                          "--batch_size instead — only C,H,W are read from it.")
parser.add_argument("--batch_size", type=int, default=1,
                     help="Number of images per inference call. Use 1 to reproduce "
                          "plain single-image inference.")
parser.add_argument("--score_threshold", type=float, default=0.3)
parser.add_argument("--max_detections", type=int, default=100)
parser.add_argument("--tegrastats_interval_ms", type=int, default=200,
                     help="Sampling interval passed to tegrastats --interval.")
parser.add_argument("--image_extensions", type=str, default=".jpg,.jpeg,.png,.bmp",
                     help="Comma-separated list of file extensions to treat as images "
                          "when --annotations is not provided (so the script knows what "
                          "to scan for in --image_dir).")
args = parser.parse_args()

engine_filename = os.path.basename(args.engine)
clean_model_name = engine_filename.split("__")[0]
pred_json_path = f"tmp_preds_{clean_model_name}_batch{args.batch_size}.json"

_, _, INPUT_H, INPUT_W = (int(x) for x in args.input_shape.split(","))
input_shape = (args.batch_size, 3, INPUT_H, INPUT_W)


def load_category_map(path):
    """Returns a dict: model output index (int) -> annotation category_id (int).
    Defaults to identity mapping if no file is given (resolved lazily once we
    know how many classes are involved, so just return None here and let
    parse_detections_batch fall back to int(raw_label) directly)."""
    if path is None:
        return None
    with open(path) as f:
        raw = json.load(f)
    return {int(k): int(v) for k, v in raw.items()}


CATEGORY_MAP = load_category_map(args.category_map)


def discover_io_tensors(engine):
    """
    Returns (input_names, output_names) read directly from the engine.
    get_tensor_mode() returns a trt.TensorIOMode — this is the only
    enum TensorRT exposes for this purpose across 8.x through current
    versions, no fallback needed.
    """
    input_names, output_names = [], []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_names.append(name)
        else:
            output_names.append(name)
    return input_names, output_names


class TegrastatsSampler:
    """
    Runs `tegrastats` as a background subprocess and samples GPU core
    utilization (the GR3D_FREQ field) on a separate thread. This is the
    only one of psutil/pynvml/tegrastats that can see GPU load at all —
    psutil only knows about CPU/system RAM, and pynvml's memory query
    isn't supported on Jetson's unified-memory architecture.
    Degrades gracefully: if tegrastats isn't found on PATH (e.g. this
    script is run on a non-Jetson dev box), GPU utilization samples are
    just empty and the rest of the benchmark proceeds unaffected.
    """
    _GR3D_RE = re.compile(r"GR3D_FREQ\s+(\d+)%")

    def __init__(self, interval_ms: int = 200):
        self.interval_ms = interval_ms
        self.available = shutil.which("tegrastats") is not None
        self._proc = None
        self._thread = None
        self._stop_flag = threading.Event()
        self.samples = []

    def start(self):
        if not self.available:
            print("ℹ️  tegrastats not found on PATH — GPU utilization will be left blank. "
                  "Expected if this isn't running on actual Jetson hardware.")
            return
        try:
            self._proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
        except Exception as e:
            print(f"⚠️  Failed to launch tegrastats ({e}) — GPU utilization will be left blank.")
            self.available = False
            return
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        for line in self._proc.stdout:
            if self._stop_flag.is_set():
                break
            match = self._GR3D_RE.search(line)
            if match:
                self.samples.append(int(match.group(1)))

    def stop(self):
        self._stop_flag.set()
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        if self._thread is not None:
            self._thread.join(timeout=2)

    def avg_and_peak(self):
        if not self.samples:
            return None, None
        return sum(self.samples) / len(self.samples), max(self.samples)


def build_image_list():
    """
    Returns a list of (image_id, file_name) tuples to iterate over.
    If --annotations was given, image_id/file_name come from the COCO-format
    json's "images" list (so predictions line up with ground truth by id).
    Otherwise, every file under --image_dir matching --image_extensions is
    used, with a synthetic sequential id (fine, since there's no ground
    truth to match ids against in this mode).
    """
    if args.annotations is not None:
        with open(args.annotations) as f:
            gt = json.load(f)
        for key in ("images", "annotations", "categories"):
            if key not in gt:
                raise ValueError(
                    f"--annotations file is missing a '{key}' key — it doesn't look like "
                    f"a COCO-format json. See this script's docstring for the expected shape."
                )
        return [(img["id"], img["file_name"]) for img in gt["images"]], gt

    exts = tuple(e.strip().lower() for e in args.image_extensions.split(","))
    files = sorted(f for f in os.listdir(args.image_dir) if f.lower().endswith(exts))
    if not files:
        raise RuntimeError(
            f"No images found in {args.image_dir} matching extensions {exts}. "
            f"Pass --image_extensions if your files use a different suffix."
        )
    return [(i, fname) for i, fname in enumerate(files)], None


def parse_detections_batch(buffers, output_names, slot_idx, orig_w, orig_h, input_w, input_h):
    if len(output_names) != 3:
        raise RuntimeError(
            f"parse_detections_batch assumes 3 outputs (labels, boxes, scores) but engine has "
            f"{len(output_names)}: {output_names}."
        )
    boxes_name = next((n for n in output_names if buffers[n].shape[-1] == 4), None)
    remaining = [n for n in output_names if n != boxes_name]
    if boxes_name is None or len(remaining) != 2:
        raise RuntimeError(
            f"Could not identify boxes/labels/scores by shape among {output_names} "
            f"(shapes: {[buffers[n].shape for n in output_names]})."
        )
    scores_name, labels_name = remaining
    if buffers[labels_name].dtype.kind == "f" and buffers[scores_name].dtype.kind != "f":
        scores_name, labels_name = labels_name, scores_name

    labels = buffers[labels_name][slot_idx]
    boxes = buffers[boxes_name][slot_idx]
    scores = buffers[scores_name][slot_idx]

    valid_ids = np.where(scores > args.score_threshold)[0]
    top_indices = valid_ids[np.argsort(scores[valid_ids])[::-1]][: args.max_detections]

    detections = []
    for k in top_indices:
        box, score, raw_label = boxes[k], float(scores[k]), int(labels[k])

        if CATEGORY_MAP is not None:
            if raw_label not in CATEGORY_MAP:
                continue  # model index has no known mapping — skip rather than guess
            category_id = CATEGORY_MAP[raw_label]
        else:
            category_id = raw_label  # identity mapping default

        xmin, ymin = box[0] * (orig_w / input_w), box[1] * (orig_h / input_h)
        bw = (box[2] - box[0]) * (orig_w / input_w)
        bh = (box[3] - box[1]) * (orig_h / input_h)
        if bw <= 0 or bh <= 0:
            continue

        detections.append({
            "category_id": category_id,
            "bbox": [float(xmin), float(ymin), float(bw), float(bh)],
            "score": score,
        })
    return detections


def main():
    tracker = OfflineEmissionsTracker(
        country_iso_code="DEU", log_level="error", output_dir="results/", output_file="carbon_raw.csv"
    )
    tracker.start()
    tegra = TegrastatsSampler(interval_ms=args.tegrastats_interval_ms)
    tegra.start()

    emissions_kg = 0.0
    coco_results = []
    batch_latency_records = []
    per_image_latency_records = []
    skipped_images = 0

    try:
        image_list, gt_dict = build_image_list()

        init_ram = psutil.virtual_memory().used / (1024 ** 2)

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(args.engine, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        cuda_stream = cuda.Stream()
        start_event, end_event = cuda.Event(), cuda.Event()

        input_names, output_names = discover_io_tensors(engine)
        if len(input_names) != 1:
            raise RuntimeError(
                f"Expected exactly 1 input tensor, found {len(input_names)}: {input_names}. "
                f"This script assumes a single-input detector."
            )
        input_name = input_names[0]
        if not output_names:
            raise RuntimeError("Engine reports zero output tensors — cannot benchmark.")
        print(f"📌 Discovered input tensor: '{input_name}'  |  output tensors: {output_names}")
        print(f"📦 Batch size: {args.batch_size}")

        # Set the input shape FIRST — output shapes for dynamic-shape engines
        # are only resolvable from the context after this, so allocation
        # reads context.get_tensor_shape() rather than the engine's static
        # get_tensor_shape().
        context.set_input_shape(input_name, input_shape)

        buffers, device_ptrs = {}, {}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = tuple(context.get_tensor_shape(name))
            if -1 in shape:
                shape = tuple(d if d != -1 else args.batch_size for d in shape)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            buffers[name] = cuda.pagelocked_empty(shape, dtype=dtype)
            device_ptrs[name] = cuda.mem_alloc(buffers[name].nbytes)
            context.set_tensor_address(name, int(device_ptrs[name]))

        # Warmup pass
        np.copyto(buffers[input_name], np.zeros(input_shape, dtype=np.float32))
        cuda.memcpy_htod_async(device_ptrs[input_name], buffers[input_name], cuda_stream)
        context.execute_async_v3(stream_handle=cuda_stream.handle)
        cuda_stream.synchronize()

        peak_ram_observed = init_ram

        for batch_start in range(0, len(image_list), args.batch_size):
            batch_items = image_list[batch_start: batch_start + args.batch_size]
            batch_input_buffer = np.zeros(input_shape, dtype=np.float32)
            batch_metadata = []  # (slot_idx, img_id, file_name, orig_w, orig_h)

            for slot_idx, (img_id, file_name) in enumerate(batch_items):
                try:
                    img_path = os.path.join(args.image_dir, file_name)
                    if not os.path.exists(img_path):
                        skipped_images += 1
                        continue
                    raw_img = cv2.imread(img_path)
                    if raw_img is None:
                        print(f"⚠️  Skipping unreadable image: {img_path}")
                        skipped_images += 1
                        continue

                    orig_h, orig_w = raw_img.shape[:2]
                    resized = cv2.resize(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB), (INPUT_W, INPUT_H))
                    input_data = np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))
                    batch_input_buffer[slot_idx] = input_data
                    batch_metadata.append((slot_idx, img_id, file_name, orig_w, orig_h))

                except Exception as e:
                    print(f"⚠️  Skipping image_id={img_id} ({file_name}) after a preprocessing error: {e}")
                    skipped_images += 1
                    continue

            if not batch_metadata:
                continue

            try:
                np.copyto(buffers[input_name], batch_input_buffer)

                start_event.record(cuda_stream)
                cuda.memcpy_htod_async(device_ptrs[input_name], buffers[input_name], cuda_stream)
                context.execute_async_v3(stream_handle=cuda_stream.handle)
                for name in output_names:
                    cuda.memcpy_dtoh_async(buffers[name], device_ptrs[name], cuda_stream)
                end_event.record(cuda_stream)
                cuda_stream.synchronize()

                batch_latency_ms = end_event.time_since(start_event)
                batch_latency_records.append(batch_latency_ms)
                per_image_latency_records.extend(
                    [batch_latency_ms / len(batch_metadata)] * len(batch_metadata)
                )

                peak_ram_observed = max(peak_ram_observed, psutil.virtual_memory().used / (1024 ** 2))

                for slot_idx, img_id, file_name, orig_w, orig_h in batch_metadata:
                    detections = parse_detections_batch(
                        buffers, output_names, slot_idx, orig_w, orig_h, INPUT_W, INPUT_H
                    )
                    for det in detections:
                        coco_results.append({"image_id": int(img_id), **det})

            except Exception as e:
                print(f"⚠️  Skipping batch starting at index={batch_start} after a processing error: {e}")
                skipped_images += len(batch_metadata)
                continue
    finally:
        try:
            emissions_kg = tracker.stop() or 0.0
        except Exception:
            pass
        tegra.stop()

    gpu_avg, gpu_peak = tegra.avg_and_peak()
    write_results(coco_results, per_image_latency_records, batch_latency_records, peak_ram_observed,
                  init_ram, emissions_kg, skipped_images, gpu_avg, gpu_peak, gt_dict)


def write_results(coco_results, per_image_latency_records, batch_latency_records, peak_ram_observed,
                   init_ram, emissions_kg, skipped_images, gpu_avg, gpu_peak, gt_dict):
    mAP, mAP_50 = 0.0, 0.0

    if gt_dict is not None and coco_results:
        # Lazy import: pycocotools is only needed in mAP mode, so a user who
        # only wants speed/telemetry numbers (no --annotations) isn't forced
        # to have it installed.
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        with open(pred_json_path, "w") as f:
            json.dump(coco_results, f)
        coco_gt = COCO(args.annotations)
        coco_dt = coco_gt.loadRes(pred_json_path)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        mAP, mAP_50 = coco_eval.stats[0], coco_eval.stats[1]
        if os.path.exists(pred_json_path):
            os.remove(pred_json_path)
    elif gt_dict is None:
        print("ℹ️  No --annotations provided — skipping mAP, recording speed/telemetry metrics only.")

    avg_batch_latency = float(np.mean(batch_latency_records)) if batch_latency_records else 0.0
    avg_per_image_latency = float(np.mean(per_image_latency_records)) if per_image_latency_records else 0.0
    throughput = (1000.0 / avg_per_image_latency) if avg_per_image_latency > 0 else 0.0
    gpu_avg_str = f"{gpu_avg:.1f}" if gpu_avg is not None else ""
    gpu_peak_str = f"{gpu_peak:.0f}" if gpu_peak is not None else ""

    os.makedirs(os.path.dirname(args.metrics_csv) or ".", exist_ok=True)
    csv_exists = os.path.exists(args.metrics_csv)
    with open(args.metrics_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow([
                "Model_Config", "Batch_Size", "Avg_Batch_Latency_ms", "Avg_Latency_ms", "Throughput_FPS",
                "Peak_Unified_RAM_Delta_MB", "Avg_GPU_Util_Pct", "Peak_GPU_Util_Pct",
                "CO2_Produced_kg", "mAP_0.5:0.95", "mAP_0.5", "Num_Detections", "Images_Skipped",
            ])
        writer.writerow([
            clean_model_name, args.batch_size, f"{avg_batch_latency:.2f}", f"{avg_per_image_latency:.2f}",
            f"{throughput:.1f}", f"{peak_ram_observed - init_ram:.1f}", gpu_avg_str, gpu_peak_str,
            f"{emissions_kg:.6f}", f"{mAP:.3f}", f"{mAP_50:.3f}", len(coco_results), skipped_images,
        ])

    print(f"📈 Telemetry written for: {clean_model_name} (batch_size={args.batch_size})  "
          f"(detections={len(coco_results)}, images_skipped={skipped_images}, "
          f"gpu_avg={gpu_avg_str or 'n/a'}%)")


if __name__ == "__main__":
    main()
