#coding:utf-8

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip
import line
import advance_lane_detec


def main():
	left_line = line.Line()
	right_line = line.Line()
	camera_cal_img_path = './camera_cal'
	LD = lane_detection.Lane_Detection(camera_cal_img_path, left_line, right_line)
	LD.get_obj_img_points()
	LD.get_M_Minv()

	# 处理文件夹里面的图片
	original_img = LD.get_images_by_dir('./test_img_one/')
	result_img = []
	for img in original_img:
		res = LD.processing(img)
		result_img.append(res)
		# plt.subplot(235)
		# plt.imshow(res)
		# plt.show()


	# 处理单个视频
	# video_input_path = 'video_input/video_input_03.mp4'
	# video_output_path = 'video_output/video_output_03.mp4'
	# project_video_clip = VideoFileClip(video_input_path)
	# project_video_out_clip = project_video_clip.fl_image(lambda clip: LD.processing(clip))
	# project_video_out_clip.write_videofile(video_output_path, audio=False)



if __name__ == '__main__':
	main()
