#!/usr/bin/env python
# coding: utf-8
# Aryan Thakur

# In[ ]:


import cv2
import numpy as np
import pyrealsense2 as rs
import torch
import logging
from ultralytics import YOLO
from networktables import NetworkTables
import cv2
import time


# In[ ]:


frame_counter = 0
start_time = time.time()
fps = 0


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cpu is painfully slow
print(f"Using device: {device}")


# In[ ]:


logging.basicConfig(level=logging.DEBUG)

NetworkTables.initialize(server='10.0.86.11')

sd = NetworkTables.getTable("depth_camera")


# In[ ]:


model = YOLO('yolov8n-seg.pt').to(device)  # or use a different model like 'yolov8.pt', i want to find a depth supported model later


# In[ ]:


pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

pipeline_profile = pipeline.start(config)

hole_filling = rs.hole_filling_filter()

depth_sensor = pipeline_profile.get_device().first_depth_sensor()

# Set high accuracy preset manually
if depth_sensor.supports(rs.option.visual_preset):
	depth_sensor.set_option(rs.option.visual_preset, 3)  # 3 is the value for High Accuracy preset
	print("High accuracy preset loaded")
else:
	print("WARNING: Device does not support visual presets")


# In[ ]:


try:
	while True:
		frames = pipeline.wait_for_frames()
		depth_frame = frames.get_depth_frame()
		color_frame = frames.get_color_frame()
		if not depth_frame or not color_frame:
			continue

		filled_depth = hole_filling.process(depth_frame) # there are some holes in the depth image, fill them

		depth_image = np.asanyarray(depth_frame.get_data())
		filled_depth_image = np.asanyarray(filled_depth.get_data())
		color_image = np.asanyarray(color_frame.get_data())

		depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
		filled_depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(filled_depth_image, alpha=0.03), cv2.COLORMAP_JET)

		color_seg = color_image.copy()
		depth_seg = depth_colormap.copy()
		filled_depth_seg = filled_depth_colormap.copy()

		results = model(color_image, device=device)

		# Process detection results
		for r in results:
			boxes = r.boxes
			masks = r.masks

			if masks is not None:
				for seg, box in zip(masks.xy, boxes):
					x1, y1, x2, y2 = map(int, box.xyxy[0])

					cv2.polylines(color_seg, [seg.astype(int)], True, (0, 255, 0), 2)
					cv2.polylines(depth_seg, [seg.astype(int)], True, (0, 255, 0), 2)
					cv2.polylines(filled_depth_seg, [seg.astype(int)], True, (0, 255, 0), 2)

					center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
					depth = filled_depth_image[center_y, center_x]
					depth_meters = depth / 1000.0  

					cv2.rectangle(color_seg, (x1, y1), (x2, y2), (255, 0, 0), 2)
					cv2.putText(color_seg, f"{r.names[int(box.cls)]} {depth_meters:.2f}m",
								(x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
					
					cv2.rectangle(depth_seg, (x1, y1), (x2, y2), (255, 0, 0), 2)
					cv2.putText(depth_seg, f"{r.names[int(box.cls)]} {depth_meters:.2f}m",
								(x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
					
					cv2.rectangle(filled_depth_seg, (x1, y1), (x2, y2), (255, 0, 0), 2)
					cv2.putText(filled_depth_seg, f"{r.names[int(box.cls)]} {depth_meters:.2f}m",
								(x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
					
					# Publish detection data to NetworkTables
					sd.putString("DetectedObject", r.names[int(box.cls)])
					sd.putNumber("ObjectDepth", depth_meters)
					sd.putNumberArray("ObjectCoordinates", [x1, y1, x2, y2])
					sd.putNumber("ObjectAngle", (center_x - 640) * 0.1) # 0.1 is the angle per pixel (horizontal field of view is 64 degrees)

					frame_counter += 1

					end_time = time.time()
					elapsed_time = end_time - start_time
					if elapsed_time > 1:  # Every second
						fps = frame_counter / elapsed_time
						print(f"FPS: {fps:.2f}")
	
						frame_counter = 0
						start_time = time.time()
	
					cv2.putText(color_seg, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

					sd.putNumber("FPS", fps)
		
					cv2.imshow('YOLOv8 Segmentation with Depth', color_seg)
					cv2.imshow('Original Depth Map with Segmentation', depth_seg)
					cv2.imshow('Hole-Filled Depth Map with Segmentation', filled_depth_seg)

					if cv2.waitKey(1) & 0xFF == ord('q'):
						break


finally:
	pipeline.stop()
	cv2.destroyAllWindows()

