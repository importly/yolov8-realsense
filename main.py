import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from ultralytics import YOLO

# Initialize RealSense camera
pipeline = rs.pipeline()
config = rs.config()

# Enable streams
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

# Start streaming
pipeline_profile = pipeline.start(config)

# Get the depth sensor
depth_sensor = pipeline_profile.get_device().first_depth_sensor()

# Set high accuracy preset manually
if depth_sensor.supports(rs.option.visual_preset):
	depth_sensor.set_option(rs.option.visual_preset, 3)  # 3 is the value for High Accuracy preset
	print("High accuracy preset loaded")
else:
	print("WARNING: Device does not support visual presets")


# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize YOLOv8 model with GPU support
model = YOLO('yolov8n-seg.pt').to(device)  # or use a different model like 'yolov8s-seg.pt'

try:
	while True:
		# Wait for a coherent pair of frames: depth and color
		frames = pipeline.wait_for_frames()
		depth_frame = frames.get_depth_frame()
		color_frame = frames.get_color_frame()
		if not depth_frame or not color_frame:
			continue

		# Convert images to numpy arrays
		depth_image = np.asanyarray(depth_frame.get_data())
		color_image = np.asanyarray(color_frame.get_data())

		# Run YOLOv8 inference on the color image
		results = model(color_image, device=device)

		# Process detection results
		for r in results:
			boxes = r.boxes
			masks = r.masks

			if masks is not None:
				for seg, box in zip(masks.xy, boxes):
					# Get bounding box coordinates
					x1, y1, x2, y2 = map(int, box.xyxy[0])

					# Draw segmentation mask
					cv2.polylines(color_image, [seg.astype(int)], True, (0, 255, 0), 2)

					# Get depth at the center of the bounding box
					center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
					depth = depth_frame.get_distance(center_x, center_y)

					# Draw bounding box and depth information
					cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
					cv2.putText(color_image, f"{r.names[int(box.cls)]} {depth:.2f}m",
					            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

		# Display the result
		cv2.imshow('YOLOv8 Segmentation with Depth', color_image)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

finally:
	# Stop streaming
	pipeline.stop()
	cv2.destroyAllWindows()