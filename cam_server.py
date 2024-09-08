#!/usr/bin/env python
# coding: utf-8
# Aryan Thakur

import cv2
import numpy as np
import pyrealsense2 as rs
import logging
from networktables import NetworkTables
import time
import sys
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Debug mode flag
DEBUG_MODE = True

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

def detect_aruco_markers(color_image):
	# Load the predefined dictionary of ArUco markers
	aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)

	# Convert the color image to grayscale for detection
	gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

	# Detect ArUco markers in the grayscale image
	corners, ids, rejected = cv2.aruco.detectMarkers(gray_image, aruco_dict)

	return corners, ids

def get_center_depth(center_x, center_y, depth_image):
	try:
		# Get the depth at the center point
		depth_value = depth_image[center_y, center_x]

		if depth_value > 0:
			return depth_value / 1000.0  # Convert depth from millimeters to meters
		else:
			return None
	except IndexError as e:
		logger.error(f"Invalid index for depth retrieval: {str(e)}")
		return None

def convert_depth_image(depth_image):
	# Normalize the depth image for display
	depth_display = cv2.convertScaleAbs(depth_image, alpha=0.03)  # Scale the depth for better visualization
	depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
	return depth_colormap

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

			# Detect ArUco markers
			corners, ids = detect_aruco_markers(color_image)

			if ids is not None:
				for i, corner in zip(ids, corners):
					# Calculate the center of the marker
					corner = corner[0]
					center_x = int(np.mean(corner[:, 0]))
					center_y = int(np.mean(corner[:, 1]))

					# Get depth at the center of the marker
					depth_meters = get_center_depth(center_x, center_y, filled_depth_image)

					if depth_meters is None:
						logger.warning(f"No valid depth data for ArUco ID {int(i)} at center [{center_x}, {center_y}]")
						continue

					# Publish ArUco detection data to NetworkTables
					sd.putNumber("ArucoID", int(i))
					sd.putNumber("ArucoDepth", depth_meters)
					sd.putNumberArray("ArucoCenter", [center_x, center_y])
					sd.putNumberArray("ArucoCorners", corner.flatten().tolist())
					sd.putNumber("ArucoAngle", (center_x - 640) * 0.1)

					# Annotate the color image in debug mode
					if DEBUG_MODE:
						# Draw the marker boundary and center
						cv2.polylines(color_image, [corner.astype(int)], True, (0, 255, 0), 2)
						cv2.circle(color_image, (center_x, center_y), 5, (255, 0, 0), -1)

						# Overlay depth and ID information on the image
						cv2.putText(color_image, f"ID: {int(i)}", (center_x - 50, center_y - 10),
						            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
						cv2.putText(color_image, f"Depth: {depth_meters:.2f}m", (center_x - 50, center_y + 10),
						            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

						logger.debug(f"Detected ArUco ID {int(i)} at depth {depth_meters:.2f}m, center: [{center_x}, {center_y}]")

			frame_counter += 1
			end_time = time.time()
			elapsed_time = end_time - start_time
			if elapsed_time > 1:
				fps = frame_counter / elapsed_time
				logger.info(f"FPS: {fps:.2f}")
				frame_counter = 0
				start_time = time.time()

			sd.putNumber("FPS", fps)

			# Show live video if in debug mode
			if DEBUG_MODE:
				# Display the color image with annotations
				cv2.imshow('ArUco Detection', color_image)

				# Display the depth data
				depth_colormap = convert_depth_image(filled_depth_image)
				cv2.imshow('Depth Data', depth_colormap)

				# Exit if 'q' key is pressed
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break

	except KeyboardInterrupt:
		logger.info("Program interrupted by user")
	except Exception as e:
		logger.error(f"An error occurred: {str(e)}")
		logger.error(traceback.format_exc())
	finally:
		pipeline.stop()
		cv2.destroyAllWindows()
		logger.info("Program terminated")

if __name__ == "__main__":
	while True:
		try:
			main()
		except Exception as e:
			logger.error(f"Main loop crashed. Restarting... Error: {str(e)}")
			time.sleep(5)  # Wait for 5 seconds before restarting
