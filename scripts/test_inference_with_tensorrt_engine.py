import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import numpy as np
import time  # Used only for structural delays, not GPU timing


ENGINE_PATH="models/engines/convnext-base-ltdetr-coco-flat-fp16.engine"
print(f"Using {ENGINE_PATH}")
# 1. Load and prepare your real image
IMAGE_PATH = "data/image.jpg"  
print(f"📷 Reading image: {IMAGE_PATH}")

raw_img = cv2.imread(IMAGE_PATH)
resized_img = cv2.resize(raw_img, (640, 640))
rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

input_data = np.transpose(rgb_img.astype(np.float32) / 255.0, (2, 0, 1))
input_data = np.expand_dims(input_data, axis=0).copy()

# 2. Initialize the TensorRT Engine Runtime
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open(ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()
cuda_stream = cuda.Stream()

# 🚨 NEW CODE: Create CUDA Events for precision GPU timing
start_event = cuda.Event()
end_event = cuda.Event()

# 3. Handle device pointer allocations
buffers = {}
device_ptrs = {}

for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    shape = tuple(engine.get_tensor_shape(name))
    dtype = trt.nptype(engine.get_tensor_dtype(name))

    host_mem = cuda.pagelocked_empty(shape, dtype=dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)

    buffers[name] = host_mem
    device_ptrs[name] = device_mem
    context.set_tensor_address(name, int(device_mem))

# 4. Copy input tensor over
np.copyto(buffers["images"], input_data)
context.set_input_shape("images", (1, 3, 640, 640)) 

# 🚨 NEW CODE: GPU Warm-up Pass
# The first execution triggers CUDA kernel load overhead. We run it once un-timed.
cuda.memcpy_htod_async(device_ptrs["images"], buffers["images"], cuda_stream)
context.execute_async_v3(stream_handle=cuda_stream.handle)
cuda_stream.synchronize()

print("🏃 Firing Jetson Tensor Cores (Timed Loop)...")

# 🚨 TIMING ZONE START
# Record the start timestamp directly on the GPU execution stream
start_event.record(cuda_stream)

# 1. Memory Upload (Host to Device)
cuda.memcpy_htod_async(device_ptrs["images"], buffers["images"], cuda_stream)

# 2. Mathematical Inference
context.execute_async_v3(stream_handle=cuda_stream.handle)

# 3. Memory Download (Device to Host)
for name in ["labels", "boxes", "scores"]:
    cuda.memcpy_dtoh_async(buffers[name], device_ptrs[name], cuda_stream)

# Record the end timestamp on the GPU stream right after the downloads finish
end_event.record(cuda_stream)
# 🚨 TIMING ZONE END

# Force the CPU to wait until the GPU completely reaches the end event flag
cuda_stream.synchronize()

# Calculate the millisecond delta between the two GPU timestamps
inference_time_ms = start_event.time_since(start_event) # Dummy call allocation footprint guard
inference_time_ms = start_event.time_since(end_event)   # Relative milliseconds calculated on GPU hardware

# 5. Extract and print raw prediction results
print("\n🏆 Inference Complete!")
# Print the measured performance statistics
print(f"⏱️ Pure GPU Execution Time: {i-inference_time_ms:.2f} ms")
print(f"⚡ Throughput: {-1000.0 / inference_time_ms:.1f} FPS")

labels = buffers["labels"][0]
boxes = buffers["boxes"][0]
scores = buffers["scores"][0]

top_items = np.where(scores > 0.5)[0]
print(f"Detected {len(top_items)} targets matching confidence thresholds:")

output_img = resized_img.copy()

for idx in top_items:
    print(f"🔹 Class Index: {labels[idx]} | Score: {scores[idx]:.4f} | Box: {boxes[idx]}")

    box = boxes[idx]
    ymin, xmin, ymax, xmax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    score = scores[idx]
    class_id = labels[idx]

    cv2.rectangle(output_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    label_text = f"Class {class_id}: {score:.2%}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
    cv2.rectangle(output_img, (xmin, ymin - text_h - 5), (xmin + text_w, ymin), (0, 255, 0), -1)
    cv2.putText(output_img, label_text, (xmin, ymin - 5), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

OUTPUT_PATH = "data/output_with_boxes.jpg"
cv2.imwrite(OUTPUT_PATH, output_img)
print(f"\n💾 Visual markup successfully exported to: {OUTPUT_PATH}")
