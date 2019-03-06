#!/usr/local/bin/python3
# -*- coding:utf-8 -*-

from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
import numpy as np
import cv2
import os

class Lane_Detection(object):
	def __init__(self):
		self.blur_ksize = 5
		self.canny_low_threshold = 70
		self.canny_high_threshold = 170
		self.rho = 1
		self.theta = np.pi / 180
		self.threshold = 15
		self.min_line_length = 40
		self.max_line_gap = 20

	def get_images_by_dir(self, dir_name):
		res_file_list = []
		img_suffix = ('.jpg', '.png', '.jpeg') # 可以使用的图像类型
		if dir_name[-1] == '/':
			dir_name = dir_name[:-1]
		img_names = os.listdir(dir_name)
		img_paths = [dir_name + '/' + img_name for img_name in img_names]
		for img in img_paths:
			for suffix in img_suffix:
				if img.endswith(suffix):
					res_file_list.append(img)
		imgs = [ mplimg.imread(path) for path in res_file_list ]
		return imgs

	def show_img(self, flag, img_list):
		plt.figure(flag)
		count = 1
		for image in img_list:
			plt.subplot(len(img_list) / 2 + 1, 2, count)
			plt.imshow(image)
			count = count + 1
		plt.show()

	def roi_mask(self, img, vertices):
		mask = np.zeros_like(img)
		if len(img.shape) > 2:
			channel_count = img.shape[2]
			mask_color = (255,) * channel_count
		else:
			mask_color = 255
			cv2.fillPoly(mask, vertices, mask_color)
			masked_img = cv2.bitwise_and(img, mask)
		return masked_img

	def draw_roi(self, img, vertices):
		cv2.polylines(img, vertices, True, [255, 0, 0], thickness=2)

	def draw_lines(self, img, lines, color=[255, 0, 0], thickness=2):
		for line in lines:
			for x1, y1, x2, y2 in line:
				cv2.line(img, (x1, y1), (x2, y2), color, thickness)

	def draw_lanes(self, img, lines, color=[255, 0, 0], thickness=1):
	    left_lines = []
	    right_lines = []
	    top_y = 1e6

	    for line in lines:
	        for x1, y1, x2, y2 in line:
	            if x1 != x2:
	                slope = (y2 - y1) / (x2 - x1)
	                if slope > 0:
	                    left_lines.append([x1, y1, x2, y2])
	                else:
	                    right_lines.append([x1, y1, x2, y2])
	            if top_y > y1:
	                top_y = y1
	            if top_y > y2:
	                top_y = y2
	    if len(left_lines) > 0:
	        left_line = [0, 0, 0, 0]
	        for line in left_lines:
	            assert (len(line) == 4)
	            for i in range(4):
	                left_line[i] += (line[i] / len(left_lines))
	        slope = (left_line[3] - left_line[1]) / (left_line[2] - left_line[0])
	        top_x = left_line[0] + (top_y - left_line[1]) / slope
	        bottom_x = left_line[0] + (img.shape[0] - left_line[1]) / slope
	        cv2.line(img, (int(bottom_x), img.shape[0]), (int(top_x), int(top_y)), color, thickness * 10)

	    if len(right_lines) > 0:
	        right_line = [0, 0, 0, 0]
	        for line in right_lines:
	            assert (len(line) == 4)
	            for i in range(4):
	                right_line[i] += (line[i] / len(right_lines))
	        slope = (right_line[3] - right_line[1]) / (right_line[2] - right_line[0])
	        top_x = right_line[0] + (top_y - right_line[1]) / slope
	        bottom_x = right_line[0] + (img.shape[0] - right_line[1]) / slope
	        cv2.line(img, (int(bottom_x), img.shape[0]), (int(top_x), int(top_y)), color, thickness * 10)

	def hough_lines(self, img, rho, theta, threshold, min_line_len, max_line_gap):
		lines = cv2.HoughLinesP(img, self.rho, self.theta, self.threshold, np.array([]), minLineLength = min_line_len, maxLineGap = max_line_gap)
		line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)#生成绘制直线的绘图板，黑底
		line_img_copy = line_img.copy()
		self.draw_lines(line_img_copy, lines)
		self.draw_lanes(line_img, lines)
		return line_img,line_img_copy

	def edge_detection(self, img):
		grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		gaussianblur = cv2.GaussianBlur(grey_img, (self.blur_ksize,self.blur_ksize),0)
		img_canny = cv2.Canny(gaussianblur, self.canny_low_threshold , self.canny_high_threshold)
		return img_canny

	def process_an_image(self, img):
		plt.subplot(2, 3, 1)
		plt.imshow(img)
		plt.axis("off")
		plt.title("original")
		edge_detec = self.edge_detection(img)
		# cv2.imshow('edge_detection', edge_detec)
		plt.subplot(2, 3, 2)
		plt.imshow(edge_detec, cmap='gray')
		plt.axis("off")
		plt.title("Edge_Detec")
		roi_vtx = np.array([[(90, img.shape[0]), (570, 470), (720, 470), (1200, img.shape[0])]])
		roi_edges = self.roi_mask(edge_detec, roi_vtx)
		# cv2.imshow('roi_edges', roi_edges)
		plt.subplot(2, 3, 3)
		plt.imshow(roi_edges, cmap='gray')
		plt.axis("off")
		plt.title("ROI_Res")
		line_img,line_img_copy = self.hough_lines(roi_edges, self.rho, self.theta, self.threshold, self.min_line_length, self.max_line_gap)
		# cv2.imshow('line_img_copy', line_img_copy)
		# cv2.imshow('line_img', line_img)
		plt.subplot(2, 3, 4)
		plt.imshow(line_img, cmap='gray')
		plt.axis("off")
		plt.title("Hough_Res")
		line_img_copy_show = cv2.addWeighted(img, 0.8, line_img_copy, 1, 0)
		plt.subplot(2, 3, 5)
		plt.imshow(line_img_copy_show, cmap='gray')
		plt.axis("off")
		plt.title("Detec_Out")
		res_img = cv2.addWeighted(img, 0.8, line_img, 1, 0)
		return res_img

if __name__ == "__main__":
	LD = Lane_Detection()
	# 处理图片
	image_paths = './test_img/'
	ori_img_list = LD.get_images_by_dir(image_paths)
	res_img = []
	for ori_img in ori_img_list:
		res = LD.process_an_image(ori_img)
		res_img.append(res)
		plt.subplot(2, 3, 6)
		plt.imshow(res)
		plt.axis("off")
		plt.title("Output")
		plt.show()

	# 处理视频
	# print("start to process the video....")
	# input_path = './video_input/straight_lane_test.mp4'
	# output_path = './video_output/output_video.mp4'
	# clip = VideoFileClip(input_path)
	# out_clip = clip.fl_image(LD.process_an_image)
	# out_clip.write_videofile(output_path, audio=False)
	# print("process the video over....")
