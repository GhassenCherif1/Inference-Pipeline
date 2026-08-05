#!/usr/bin/env python3
# scripts/benchmark_batch_coco_detection.py
"""
Benchmarks one compiled TensorRT engine against the COCO val2017 set with
configurable batch size: latency, throughput, peak RAM delta, GPU
utilization, CO2 (CodeCarbon), and mAP.

This is the batched sibling of benchmark_coco_object_detection.py. All
engine-loading, tensor discovery, telemetry (tegrastats, CodeCarbon, RAM
tracking), error handling, and CSV output are shared so results from both
scripts land in the same metrics file and compare directly — the only
difference is that inference runs --batch_size images at a time instead of
one at a time, which is what you want when characterizing real deployment
throughput rather than single-image latency.

Jetson-Specific Notes:
  - pynvml is not used. Jetson Orin Nano's Unified Memory Architecture
    (UMA) means there's no separate VRAM pool for NVML to report on, so
    nvmlDeviceGetMemoryInfo raises NotSupported. psutil's system RAM
    reading already captures the same unified pool.
  - GPU core utilization (the one signal psutil/pynvml can't provide
    here) comes from `tegrastats`, NVIDIA's own Tegra/Jetson monitoring
    tool, run as a background subprocess and sampled on a separate
    thread for the duration of the inference loop. If tegrastats isn't
    on PATH (e.g. running this on a non-Jetson dev box), GPU utilization
    is simply left blank rather than crashing the benchmark.
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
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

parser = argparse.ArgumentParser()
parser.add_argument("--engine", type=str, required=True, help="Path to compiled stamped .engine binary")
parser.add_argument("--annotations", type=str, default="data/coco_dataset/annotations/instances_val2017.json")
parser.add_argument("--image_dir", type=str, default="data/coco_dataset/images/val2017/")
parser.add_argument("--metrics_csv", type=str, default="results/coco_detection_batch.csv")
parser.add_argument("--input_shape", type=str, default="1,3,640,640",
                     help="Comma-separated NCHW shape. The N here is ignored — N is set from "
                          "--batch_size instead — only C,H,W are read from it.")
parser.add_argument("--batch_size", type=int, default=1,
                     help="Number of images per inference call. Use 1 to reproduce "
                          "single-image behaviour identical to benchmark_coco_object_detection.py.")
parser.add_argument("--score_threshold", type=float, default=0.3)
parser.add_argument("--max_detections", type=int, default=100)
parser.add_argument("--tegrastats_interval_ms", type=int, default=200,
                     help="Sampling interval passed to tegrastats --interval.")
args = parser.parse_args()

engine_filename = os.path.basename(args.engine)

# "vitt16_int8_entropy_fp16_safeMHA__trt10.3-sm87.engine" -> "vitt16_int8_entropy_fp16_safeMHA"
clean_model_name = engine_filename.split("__")[0]
pred_json_path = f"tmp_preds_{clean_model_name}_batch{args.batch_size}.json"

_, _, INPUT_H, INPUT_W = (int(x) for x in args.input_shape.split(","))
input_shape = (args.batch_size, 3, INPUT_H, INPUT_W)

COCO_IDX_TO_CAT_ID = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27,
    28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
    54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80,
    81, 82, 84, 85, 86, 87, 88, 89, 90,
]


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

    # Matches "GR3D_FREQ 41%@611" (JP6-style, single value) or
    # "GR3D_FREQ 0%@114 GR3D2_FREQ 0%@114" (older dual-core style) —
    # only the first GR3D_FREQ percentage is captured either way.
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
        # Establish baseline RAM usage (captures the whole unified memory pool on Jetson)
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
        # are only resolvable from the context after this, which is why
        # allocation reads context.get_tensor_shape() rather than the
        # engine's static get_tensor_shape() used in the single-image script.
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

        coco_gt = COCO(args.annotations)
        image_ids = coco_gt.getImgIds()

        # Warmup pass
        np.copyto(buffers[input_name], np.zeros(input_shape, dtype=np.float32))
        cuda.memcpy_htod_async(device_ptrs[input_name], buffers[input_name], cuda_stream)
        context.execute_async_v3(stream_handle=cuda_stream.handle)
        cuda_stream.synchronize()

        peak_ram_observed = init_ram

        for batch_start in range(0, len(image_ids), args.batch_size):
            batch_img_ids = image_ids[batch_start: batch_start + args.batch_size]
            batch_input_buffer = np.zeros(input_shape, dtype=np.float32)
            batch_metadata = []  # (slot_idx, img_id, orig_w, orig_h)

            for slot_idx, img_id in enumerate(batch_img_ids):
                try:
                    img_info = coco_gt.loadImgs(img_id)[0]
                    img_path = os.path.join(args.image_dir, img_info["file_name"])
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
                    batch_metadata.append((slot_idx, img_id, orig_w, orig_h))

                except Exception as e:
                    print(f"⚠️  Skipping image_id={img_id} after a preprocessing error: {e}")
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

                # Track global system RAM peak (encompasses unified memory shifts)
                peak_ram_observed = max(peak_ram_observed, psutil.virtual_memory().used / (1024 ** 2))

                for slot_idx, img_id, orig_w, orig_h in batch_metadata:
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
                  init_ram, emissions_kg, skipped_images, gpu_avg, gpu_peak)


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
        if raw_label < 0 or raw_label >= len(COCO_IDX_TO_CAT_ID):
            continue
        xmin, ymin = box[0] * (orig_w / input_w), box[1] * (orig_h / input_h)
        bw = (box[2] - box[0]) * (orig_w / input_w)
        bh = (box[3] - box[1]) * (orig_h / input_h)
        if bw <= 0 or bh <= 0:
            continue
        detections.append({
            "category_id": COCO_IDX_TO_CAT_ID[raw_label],
            "bbox": [float(xmin), float(ymin), float(bw), float(bh)],
            "score": score,
        })
    return detections


def write_results(coco_results, per_image_latency_records, batch_latency_records, peak_ram_observed,
                   init_ram, emissions_kg, skipped_images, gpu_avg, gpu_peak):
    mAP, mAP_50 = 0.0, 0.0
    if coco_results:
        coco_gt = COCO(args.annotations)
        with open(pred_json_path, "w") as f:
            json.dump(coco_results, f)
        coco_dt = coco_gt.loadRes(pred_json_path)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        mAP, mAP_50 = coco_eval.stats[0], coco_eval.stats[1]
        if os.path.exists(pred_json_path):
            os.remove(pred_json_path)

    avg_batch_latency = float(np.mean(batch_latency_records)) if batch_latency_records else 0.0
    avg_per_image_latency = float(np.mean(per_image_latency_records)) if per_image_latency_records else 0.0
    throughput = (1000.0 / avg_per_image_latency) if avg_per_image_latency > 0 else 0.0
    gpu_avg_str = f"{gpu_avg:.1f}" if gpu_avg is not None else ""
    gpu_peak_str = f"{gpu_peak:.0f}" if gpu_peak is not None else ""

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
