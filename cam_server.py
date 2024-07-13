#!/usr/bin/env python
# coding: utf-8
# Aryan Thakur

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
import logging
from ultralytics import YOLO
from networktables import NetworkTables
import time
import sys
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Debug mode flag
DEBUG_MODE = False


def initialize_network_tables():
	try:
		NetworkTables.initialize(server='10.0.86.11')
		return NetworkTables.getTable("depth_camera")
	except Exception as e:
		logger.error(f"Failed to initialize NetworkTables: {str(e)}")
		return None


def initialize_camera():
	try:
		pipeline = rs.pipeline()
		config = rs.config()
		config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
		config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
		pipeline_profile = pipeline.start(config)

		hole_filling = rs.hole_filling_filter()
		depth_sensor = pipeline_profile.get_device().first_depth_sensor()

		if depth_sensor.supports(rs.option.visual_preset):
			depth_sensor.set_option(rs.option.visual_preset, 3)
			logger.info("High accuracy preset loaded")
		else:
			logger.warning("Device does not support visual presets")

		return pipeline, hole_filling
	except Exception as e:
		logger.error(f"Failed to initialize camera: {str(e)}")
		return None, None


def load_model():
	try:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		logger.info(f"Using device: {device}")
		return YOLO('yolov8n-seg.pt').to(device), device
	except Exception as e:
		logger.error(f"Failed to load model: {str(e)}")
		return None, None


def main():
	global DEBUG_MODE

	if len(sys.argv) > 1 and sys.argv[1] == "--debug":
		DEBUG_MODE = True
		logger.setLevel(logging.DEBUG)
		logger.debug("Debug mode enabled")

	sd = initialize_network_tables()
	if sd is None:
		return

	pipeline, hole_filling = initialize_camera()
	if pipeline is None or hole_filling is None:
		return

	model, device = load_model()
	if model is None or device is None:
		return

	frame_counter = 0
	start_time = time.time()
	fps = 0

	try:
		while True:
			frames = pipeline.wait_for_frames()
			depth_frame = frames.get_depth_frame()
			color_frame = frames.get_color_frame()
			if not depth_frame or not color_frame:
				continue

			filled_depth = hole_filling.process(depth_frame)

			depth_image = np.asanyarray(depth_frame.get_data())
			filled_depth_image = np.asanyarray(filled_depth.get_data())
			color_image = np.asanyarray(color_frame.get_data())

			depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
			filled_depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(filled_depth_image, alpha=0.03),
			                                          cv2.COLORMAP_JET)

			results = model(color_image, device=device)

			for r in results:
				boxes = r.boxes
				masks = r.masks

				if masks is not None:
					for seg, box in zip(masks.xy, boxes):
						x1, y1, x2, y2 = map(int, box.xyxy[0])

						center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
						depth = filled_depth_image[center_y, center_x]
						depth_meters = depth / 1000.0

						# Publish detection data to NetworkTables
						sd.putString("DetectedObject", r.names[int(box.cls)])
						sd.putNumber("ObjectDepth", depth_meters)
						sd.putNumberArray("ObjectCoordinates", [x1, y1, x2, y2])
						sd.putNumber("ObjectAngle", (center_x - 640) * 0.1)

						if DEBUG_MODE:
							logger.debug(
								f"Detected {r.names[int(box.cls)]} at depth {depth_meters:.2f}m, coordinates: [{x1}, {y1}, {x2}, {y2}]")

			frame_counter += 1
			end_time = time.time()
			elapsed_time = end_time - start_time
			if elapsed_time > 1:
				fps = frame_counter / elapsed_time
				logger.info(f"FPS: {fps:.2f}")
				frame_counter = 0
				start_time = time.time()

			sd.putNumber("FPS", fps)

	except KeyboardInterrupt:
		logger.info("Program interrupted by user")
	except Exception as e:
		logger.error(f"An error occurred: {str(e)}")
		logger.error(traceback.format_exc())
	finally:
		pipeline.stop()
		logger.info("Program terminated")


if __name__ == "__main__":
	while True:
		try:
			main()
		except Exception as e:
			logger.error(f"Main loop crashed. Restarting... Error: {str(e)}")
			time.sleep(5)  # Wait for 5 seconds before restarting
