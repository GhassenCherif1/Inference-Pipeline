#!/usr/bin/env python3
# scripts/benchmark_segmentation.py
"""
Benchmarks one compiled TensorRT semantic segmentation engine (DINOv3-EoMT
via lightly-train): latency, throughput, peak RAM delta, GPU utilization,
CO2 (CodeCarbon), and mIoU.

Works against COCO-Stuff by default, or any custom dataset that provides
ground truth as single-channel indexed PNG masks (the same format
COCO-Stuff ships, and the same format most segmentation labeling tools
export to).

----------------------------------------------------------------------
ENGINE OUTPUT ASSUMPTIONS (confirmed against an actual exported model)
----------------------------------------------------------------------
  - Input:  "images"  (N, 3, H, W) float32
  - Output: "masks"   (N, H, W)    int32  — per-pixel predicted class
                                             index, in the model's
                                             INTERNAL index space
                                             (0..num_classes-1)
            "logits"  (N, num_classes, H, W) float32 — raw per-class
                                             scores (not required for
                                             mIoU, but read off the
                                             engine for completeness /
                                             future use, e.g. confidence
                                             thresholding)

  The model's internal class indices are NOT the same as the dataset's
  real category ids (e.g. COCO-Stuff's 1-182 with gaps). Use
  --category_map to supply the translation — see lightly-train's
  `model.internal_class_to_class` tensor, which gives this directly:

      python3 -c "
      import json
      import lightly_train
      model = lightly_train.load_model('dinov3/vitl16-eomt-coco', device='cpu')
      mapping = {str(i): v for i, v in enumerate(model.internal_class_to_class.tolist())
                 if v != model.class_ignore_index}
      json.dump(mapping, open('seg_category_map.json', 'w'), indent=2)
      "

  If you don't pass --category_map, an identity mapping is assumed
  (internal index i -> ground truth class id i). This is WRONG for the
  lightly-train COCO-Stuff checkpoints (their internal index space has
  gaps relative to COCO-Stuff's real ids) — pass the map for those.

----------------------------------------------------------------------
GROUND TRUTH FORMAT
----------------------------------------------------------------------
  One single-channel PNG per image, same base filename as the image
  (e.g. image: 000000000139.jpg -> mask: 000000000139.png), where each
  pixel's value is the integer class id. This matches COCO-Stuff's
  "stuffthingmaps" download directly: 0-181 are real classes, 255 is
  the official COCO-Stuff "unlabeled" sentinel.

  --gt_ignore_value (default 255) is excluded from mIoU on the ground
  truth side. Any model-predicted pixel that doesn't appear in
  --category_map's value set is also excluded automatically (defensive
  handling for whatever the model's own ignore-index convention turns
  out to be at inference time — see this script's accompanying
  discussion for why this is handled defensively rather than assumed).

----------------------------------------------------------------------
WHAT'S SHARED WITH benchmark_node.py / benchmark_custom.py
----------------------------------------------------------------------
  - TensorRT engine loading, I/O tensor discovery, inference loop
  - tegrastats-based GPU utilization sampling (Jetson-specific)
  - psutil-based unified RAM delta tracking
  - CodeCarbon emissions tracking
  - Same CSV columns where they apply (latency/throughput/RAM/GPU/CO2),
    with mIoU/mIoU_50 replacing mAP/mAP_50, and a per-class IoU dump
    written alongside as a second file for diagnostics.
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
parser.add_argument("--mask_dir", type=str, default=None,
                     help="Optional folder of single-channel indexed PNG ground-truth masks, "
                          "one per image with matching base filename. If omitted, mIoU is "
                          "skipped and only speed/telemetry metrics are recorded.")
parser.add_argument("--category_map", type=str, default=None,
                     help="Optional path to a JSON file mapping model internal class index "
                          "(string key) -> ground-truth class id (int value). If omitted, an "
                          "identity mapping is assumed (index i -> class id i) — this is WRONG "
                          "for lightly-train's COCO-Stuff checkpoints, see this script's "
                          "docstring for how to generate the correct file.")
parser.add_argument("--num_classes", type=int, default=None,
                     help="Number of real classes for mIoU averaging. If omitted, inferred as "
                          "the number of entries in --category_map, or from the engine's "
                          "'logits' output channel count if no category_map is given.")
parser.add_argument("--gt_ignore_value", type=int, default=255,
                     help="Ground-truth pixel value to exclude from mIoU (COCO-Stuff's "
                          "'unlabeled' sentinel is 255).")
parser.add_argument("--metrics_csv", type=str, default="results/segmentation_telemetry.csv")
parser.add_argument("--per_class_iou_json", type=str, default=None,
                     help="Optional path to dump per-class IoU as JSON, for diagnostics. "
                          "Defaults to <metrics_csv directory>/per_class_iou_<model_name>.json")
parser.add_argument("--input_shape", type=str, default="1,3,512,512",
                     help="Comma-separated NCHW shape to set on the engine's input tensor. "
                          "Default matches the confirmed DINOv3-EoMT export (512x512).")
parser.add_argument("--tegrastats_interval_ms", type=int, default=200,
                     help="Sampling interval passed to tegrastats --interval.")
parser.add_argument("--image_extensions", type=str, default=".jpg,.jpeg,.png,.bmp",
                     help="Comma-separated list of file extensions to treat as images.")
parser.add_argument("--max_images", type=int, default=None,
                     help="If set, only process the first N images. Useful for a quick "
                          "sanity check before committing to a full run.")
args = parser.parse_args()

engine_filename = os.path.basename(args.engine)
clean_model_name = engine_filename.split("__")[0]
input_shape = tuple(int(x) for x in args.input_shape.split(","))
_, _, INPUT_H, INPUT_W = input_shape


def load_category_map(path):
    """Returns dict: model internal index (int) -> ground-truth class id (int).
    Returns None if no file given, signaling "use identity mapping" to callers."""
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


class ConfusionAccumulator:
    """
    Accumulates a (num_classes x num_classes) confusion matrix across the
    whole dataset, then derives per-class IoU and mIoU at the end. This is
    the standard streaming approach for segmentation mIoU — avoids holding
    every prediction/ground-truth pair in memory simultaneously, which
    matters since masks are full H*W arrays per image, not small per-image
    scalars like detection mAP's box list.
    """
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray):
        """pred, gt: int arrays of the same shape, already remapped into
        ground-truth class id space and restricted to [0, num_classes).
        valid_mask: bool array, True where the pixel should count at all
        (i.e. not ignore-index on either side)."""
        # IMPORTANT: cast to int64 BEFORE arithmetic. gt commonly arrives as
        # uint8 (cv2.imread's native dtype for single-channel PNGs), and
        # `uint8_array * python_int` keeps uint8 dtype and silently WRAPS
        # AROUND at 256 -- e.g. uint8(180) * 182 wraps to 248 instead of
        # 32760. This was scattering nearly every pixel into the wrong
        # confusion-matrix cell despite predictions/ground-truth themselves
        # being correct, which is why per-image pixel-match checks (plain
        # == comparisons, no multiplication) looked fine while mIoU stayed
        # near zero. Casting first avoids the overflow entirely.
        p = pred[valid_mask].astype(np.int64)
        g = gt[valid_mask].astype(np.int64)
        idx = g * self.num_classes + p
        counts = np.bincount(idx, minlength=self.num_classes * self.num_classes)
        self.matrix += counts.reshape(self.num_classes, self.num_classes)

    def per_class_iou(self):
        tp = np.diag(self.matrix)
        fp = self.matrix.sum(axis=0) - tp
        fn = self.matrix.sum(axis=1) - tp
        denom = tp + fp + fn
        iou = np.full(self.num_classes, np.nan)
        nonzero = denom > 0
        iou[nonzero] = tp[nonzero] / denom[nonzero]
        return iou

    def miou(self):
        iou = self.per_class_iou()
        valid = ~np.isnan(iou)
        if not valid.any():
            return 0.0
        return float(np.nanmean(iou))


def build_image_list():
    exts = tuple(e.strip().lower() for e in args.image_extensions.split(","))
    files = sorted(f for f in os.listdir(args.image_dir) if f.lower().endswith(exts))
    if not files:
        raise RuntimeError(
            f"No images found in {args.image_dir} matching extensions {exts}. "
            f"Pass --image_extensions if your files use a different suffix."
        )
    if args.max_images is not None:
        files = files[:args.max_images]
        print(f"ℹ️  --max_images set: using {len(files)} of the available images.")
    return files


def resize_short_side_and_center_crop(img, target_size, interpolation):
    """
    Resizes so the SHORT side becomes `target_size` (preserving aspect ratio,
    matching lightly-train's own model.predict() preprocessing: 'resize such
    that the short side of the image matches the crop size'), then
    center-crops to exactly (target_size, target_size).

    This must be called with IDENTICAL arguments on both the input image and
    its corresponding ground-truth mask, so the same region of the original
    image maps to the same pixels in both -- a naive independent resize of
    each to a fixed square (the previous approach) distorts aspect ratio and
    misaligns prediction vs. ground truth for any non-square source image.

    Returns the transformed array plus the crop offsets (top, left) and the
    pre-crop resized dimensions, in case downstream code needs to map back to
    original image coordinates (not needed for this script's mIoU computation,
    but kept for transparency/debugging).
    """
    h, w = img.shape[:2]
    short_side = min(h, w)
    scale = target_size / short_side
    new_h, new_w = round(h * scale), round(w * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    top = (new_h - target_size) // 2
    left = (new_w - target_size) // 2
    cropped = resized[top:top + target_size, left:left + target_size]

    # Guard against off-by-one rounding leaving the crop short of target_size
    # on one edge (can happen with odd dimensions) -- pad back up if needed.
    pad_h = target_size - cropped.shape[0]
    pad_w = target_size - cropped.shape[1]
    if pad_h > 0 or pad_w > 0:
        if cropped.ndim == 2:
            cropped = np.pad(cropped, ((0, max(pad_h, 0)), (0, max(pad_w, 0))), mode="edge")
        else:
            cropped = np.pad(cropped, ((0, max(pad_h, 0)), (0, max(pad_w, 0)), (0, 0)), mode="edge")

    return cropped, (top, left), (new_h, new_w)


def find_matching_mask(image_filename):
    """Looks for a ground-truth PNG with the same base filename as the image,
    e.g. 000000000139.jpg -> 000000000139.png, inside --mask_dir."""
    base = os.path.splitext(image_filename)[0]
    candidate = os.path.join(args.mask_dir, base + ".png")
    return candidate if os.path.exists(candidate) else None


def remap_predictions(pred_internal: np.ndarray, valid_gt_ids: set):
    """
    Converts model-internal class indices to ground-truth class id space
    using CATEGORY_MAP (or identity mapping if none given), and produces a
    validity mask marking pixels where the prediction has no known mapping
    (defensive handling for the model's own ignore-index convention,
    whatever it turns out to be at inference time, plus any internal index
    a category_map simply doesn't cover).
    """
    if CATEGORY_MAP is None:
        # Identity mapping: internal index IS the ground truth class id.
        valid = np.isin(pred_internal, list(valid_gt_ids))
        return pred_internal, valid

    # Vectorized lookup via a dense array indexed by internal class id.
    max_internal = int(pred_internal.max()) if pred_internal.size else 0
    lut_size = max(max_internal + 1, max(CATEGORY_MAP.keys(), default=-1) + 1)
    lut = np.full(lut_size, -1, dtype=np.int64)
    for internal_idx, gt_id in CATEGORY_MAP.items():
        if internal_idx < lut_size:
            lut[internal_idx] = gt_id

    safe_pred = np.clip(pred_internal, 0, lut_size - 1)
    remapped = lut[safe_pred]
    valid = remapped != -1
    return remapped, valid


def main():
    tracker = OfflineEmissionsTracker(
        country_iso_code="DEU", log_level="error", output_dir="results/", output_file="carbon_raw.csv"
    )
    tracker.start()
    tegra = TegrastatsSampler(interval_ms=args.tegrastats_interval_ms)
    tegra.start()

    emissions_kg = 0.0
    latency_records = []
    skipped_images = 0
    compute_miou = args.mask_dir is not None

    confusion = None  # built lazily once we know num_classes

    try:
        image_files = build_image_list()
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
                f"Expected exactly 1 input tensor, found {len(input_names)}: {input_names}."
            )
        input_name = input_names[0]
        if "logits" not in output_names:
            raise RuntimeError(
                f"Expected an output tensor named 'logits', found: {output_names}. "
                f"This script computes the per-pixel class prediction via argmax over "
                f"'logits' -- 'masks' is read off the engine but not used for class "
                f"prediction (confirmed it's not a class-index map for this model)."
            )
        print(f"📌 Discovered input tensor: '{input_name}'  |  output tensors: {output_names}")

        buffers, device_ptrs = {}, {}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = tuple(engine.get_tensor_shape(name))
            if -1 in shape:
                shape = tuple(d if d != -1 else 1 for d in shape)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            buffers[name] = cuda.pagelocked_empty(shape, dtype=dtype)
            device_ptrs[name] = cuda.mem_alloc(buffers[name].nbytes)
            context.set_tensor_address(name, int(device_ptrs[name]))
        context.set_input_shape(input_name, input_shape)

        # Determine num_classes for the confusion matrix, in priority order:
        # explicit --num_classes > size of --category_map's value range > logits channel count.
        if args.num_classes is not None:
            num_classes = args.num_classes
        elif CATEGORY_MAP is not None:
            num_classes = max(CATEGORY_MAP.values()) + 1
        elif "logits" in output_names:
            num_classes = buffers["logits"].shape[1]
        else:
            raise RuntimeError(
                "Could not determine --num_classes: pass it explicitly, or provide "
                "--category_map, or ensure the engine has a 'logits' output."
            )
        print(f"📊 mIoU will be computed over {num_classes} ground-truth classes"
              if compute_miou else "ℹ️  No --mask_dir provided — skipping mIoU, "
                                    "recording speed/telemetry metrics only.")
        if compute_miou:
            confusion = ConfusionAccumulator(num_classes)
        valid_gt_ids = set(range(num_classes))

        # Warmup pass
        np.copyto(buffers[input_name], np.zeros(input_shape, dtype=np.float32))
        cuda.memcpy_htod_async(device_ptrs[input_name], buffers[input_name], cuda_stream)
        context.execute_async_v3(stream_handle=cuda_stream.handle)
        cuda_stream.synchronize()

        peak_ram_observed = init_ram

        total_images = len(image_files)
        for img_counter, file_name in enumerate(image_files):
            if img_counter % 50 == 0:
                print(f"  ... {img_counter}/{total_images} images processed", flush=True)
            try:
                img_path = os.path.join(args.image_dir, file_name)
                raw_img = cv2.imread(img_path)
                if raw_img is None:
                    print(f"⚠️  Skipping unreadable image: {img_path}")
                    skipped_images += 1
                    continue

                orig_h, orig_w = raw_img.shape[:2]
                rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
                # Match model.predict()'s actual preprocessing: resize so the SHORT
                # side equals the input size (preserving aspect ratio), then center
                # crop to a square -- NOT a naive stretch to (INPUT_W, INPUT_H), which
                # distorts geometry and misaligns predictions vs. ground truth for any
                # non-square source image (confirmed empirically: 0% pixel agreement
                # with the naive-stretch approach on a 426x640 COCO image).
                assert INPUT_H == INPUT_W, (
                    "resize_short_side_and_center_crop assumes a square model input; "
                    f"got INPUT_H={INPUT_H}, INPUT_W={INPUT_W}."
                )
                resized, crop_offset, pre_crop_size = resize_short_side_and_center_crop(
                    rgb_img, INPUT_H, interpolation=cv2.INTER_LINEAR
                )
                input_data = np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))
                np.copyto(buffers[input_name], np.expand_dims(input_data, axis=0).copy())

                start_event.record(cuda_stream)
                cuda.memcpy_htod_async(device_ptrs[input_name], buffers[input_name], cuda_stream)
                context.execute_async_v3(stream_handle=cuda_stream.handle)
                for name in output_names:
                    cuda.memcpy_dtoh_async(buffers[name], device_ptrs[name], cuda_stream)
                end_event.record(cuda_stream)
                cuda_stream.synchronize()

                latency_records.append(end_event.time_since(start_event))
                peak_ram_observed = max(peak_ram_observed, psutil.virtual_memory().used / (1024 ** 2))

                if compute_miou:
                    mask_path = find_matching_mask(file_name)
                    if mask_path is None:
                        skipped_images += 1
                        continue

                    gt_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                    if gt_mask is None:
                        print(f"⚠️  Skipping unreadable mask: {mask_path}")
                        skipped_images += 1
                        continue

                    # NOTE: "masks" is NOT a class-index map for this model -- confirmed
                    # empirically it's unrelated to the class logits (0% agreement with
                    # argmax(logits)) and ranges beyond num_classes (up to num_queries-1).
                    # The real per-pixel class prediction is argmax over "logits".
                    logits = np.squeeze(buffers["logits"])  # (num_classes, H, W)
                    pred_internal = np.argmax(logits, axis=0)  # (H, W)
                    # Apply the SAME resize-short-side + center-crop transform used on
                    # the input image, with the SAME target size, so ground truth lines
                    # up pixel-for-pixel with what the model actually saw. Using a
                    # different/independent resize here (e.g. stretching gt_mask
                    # directly to pred_internal.shape) is exactly what caused near-0%
                    # pixel agreement before -- the two must go through identical
                    # geometry, not just end up at the same final shape.
                    gt_resized, _, _ = resize_short_side_and_center_crop(
                        gt_mask, INPUT_H, interpolation=cv2.INTER_NEAREST
                    )

                    pred_remapped, pred_valid = remap_predictions(pred_internal, valid_gt_ids)
                    gt_valid = gt_resized != args.gt_ignore_value
                    valid_mask = pred_valid & gt_valid

                    gt_safe = np.clip(gt_resized, 0, num_classes - 1)
                    pred_safe = np.clip(pred_remapped, 0, num_classes - 1)
                    confusion.update(pred_safe, gt_safe, valid_mask)
            except Exception as e:
                print(f"⚠️  Skipping {file_name} after a processing error: {e}")
                skipped_images += 1
                continue
    finally:
        try:
            emissions_kg = tracker.stop() or 0.0
        except Exception:
            pass
        tegra.stop()

    gpu_avg, gpu_peak = tegra.avg_and_peak()
    write_results(latency_records, peak_ram_observed, init_ram, emissions_kg,
                  skipped_images, gpu_avg, gpu_peak, confusion)


def write_results(latency_records, peak_ram_observed, init_ram, emissions_kg,
                   skipped_images, gpu_avg, gpu_peak, confusion):
    miou = 0.0
    per_class = None
    if confusion is not None:
        miou = confusion.miou()
        per_class = confusion.per_class_iou()

        dump_path = args.per_class_iou_json or os.path.join(
            os.path.dirname(args.metrics_csv) or ".", f"per_class_iou_{clean_model_name}.json"
        )
        os.makedirs(os.path.dirname(dump_path) or ".", exist_ok=True)
        with open(dump_path, "w") as f:
            json.dump({str(i): (None if np.isnan(v) else float(v)) for i, v in enumerate(per_class)},
                       f, indent=2)
        print(f"📝 Per-class IoU written to {dump_path}")

    avg_latency = float(np.mean(latency_records)) if latency_records else 0.0
    throughput = (1000.0 / avg_latency) if avg_latency > 0 else 0.0
    gpu_avg_str = f"{gpu_avg:.1f}" if gpu_avg is not None else ""
    gpu_peak_str = f"{gpu_peak:.0f}" if gpu_peak is not None else ""

    os.makedirs(os.path.dirname(args.metrics_csv) or ".", exist_ok=True)
    csv_exists = os.path.exists(args.metrics_csv)
    with open(args.metrics_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow([
                "Model_Config", "Avg_Latency_ms", "Throughput_FPS",
                "Peak_Unified_RAM_Delta_MB", "Avg_GPU_Util_Pct", "Peak_GPU_Util_Pct",
                "CO2_Produced_kg", "mIoU", "Images_Skipped",
            ])
        writer.writerow([
            clean_model_name, f"{avg_latency:.2f}", f"{throughput:.1f}",
            f"{peak_ram_observed - init_ram:.1f}", gpu_avg_str, gpu_peak_str,
            f"{emissions_kg:.6f}", f"{miou:.4f}", skipped_images,
        ])

    print(f"📈 Telemetry written for: {clean_model_name}  "
          f"(mIoU={miou:.4f}, images_skipped={skipped_images}, gpu_avg={gpu_avg_str or 'n/a'}%)")


if __name__ == "__main__":
    main()
