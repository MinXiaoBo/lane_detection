#!/usr/local/bin/python3
#coding:utf-8

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Lane_Detection(object):

	def __init__(self, camera_cal_img_path, left_line, right_line):
		self.camera_cal_img_path = camera_cal_img_path
		self.Affine_src = np.float32([[(200, 720), (585, 470), (695, 470), (1120, 720)]])
		self.Affine_dst = np.float32([[(320, 720), (320, 0), (960, 0), (960, 720)]])
		self.left_line = left_line
		self.right_line = right_line
		self.object_points = []
		self.img_points = []
		self.M = []
		self.Minv = []

	def thresholding(self, img):
		x_thresh = self.abs_sobel_thresh(img, orient='x', thresh_min=10 ,thresh_max=230)
		mag_thresh = self.mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 150))
		dir_thresh = self.dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3))
		hls_thresh = self.hls_select(img, thresh=(180, 255))
		lab_thresh = self.lab_select(img, thresh=(155, 200))
		luv_thresh = self.luv_select(img, thresh=(225, 255))
		threshholded = np.zeros_like(x_thresh)
		threshholded[((x_thresh == 1) & (mag_thresh == 1)) | ((dir_thresh == 1) & (hls_thresh == 1)) | (lab_thresh == 1) | (luv_thresh == 1)] = 1
		return threshholded

	def canny(self, img):
		kernel_size = 5
		low_threshold = 50
		high_threshold = 150
		grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		gauss_gray = cv2.GaussianBlur(grey, (kernel_size, kernel_size), 0)
		canny_edges = cv2.Canny(grey, low_threshold, high_threshold)
		return canny_edges

	def processing(self, img):
		calibrated = self.camera_calibrate(img)
		plt.subplot(231)
		plt.imshow(calibrated)
		plt.axis("off")
		thresholded = self.thresholding(calibrated)
		plt.subplot(232)
		plt.imshow(thresholded, cmap ='gray')
		plt.axis("off")
		thresholded_wraped = cv2.warpPerspective(thresholded, self.M, img.shape[1::-1], flags=cv2.INTER_LINEAR)
		plt.subplot(233)
		plt.imshow(thresholded_wraped,cmap ='gray')
		plt.axis("off")
		if self.left_line.detected and self.right_line.detected:
			left_fit, right_fit, left_lane_inds, right_lane_inds,out_img,left_fitx,right_fitx,ploty = self.find_line_by_previous(thresholded_wraped, self.left_line.current_fit, self.right_line.current_fit)
		else:
			left_fit, right_fit, left_lane_inds, right_lane_inds,out_img,left_fitx,right_fitx,ploty = self.find_line(thresholded_wraped)
		self.left_line.update(left_fit)
		self.right_line.update(right_fit)
		newwarp,area_img = self.draw_area(img, thresholded_wraped, self.Minv, left_fit, right_fit)
		plt.subplot(234)
		plt.imshow(out_img)
		plt.axis("off")
		plt.plot(left_fitx, ploty, color='yellow')
		plt.plot(right_fitx, ploty, color='yellow')
		plt.xlim(0, 1280)
		plt.ylim(720, 0)
		plt.subplot(235)
		plt.imshow(newwarp)
		plt.axis("off")
		plt.subplot(236)
		plt.imshow(area_img)
		plt.axis("off")
		plt.show()
		return area_img

	def show_img(self, img_list, flag = 'img_show'):
		plt.figure(flag)
		count = 1
		for image in img_list:
			plt.subplot(len(img_list)/3+1,3,count)
			plt.imshow(image)
			count = count + 1
		plt.show()

	def get_images_by_dir(self, dir_name):
		res_file_list = []
		img_suffix = ('.jpg', '.png', '.jpeg')  # 可以使用的图像类型
		if dir_name[-1] == '/':					# 如果路径结尾有斜杠就把他去掉 如果严谨一点可以判断两次
			dir_name = dir_name[:-1]
		img_names = os.listdir(dir_name)
		img_paths = [dir_name + '/' + img_name for img_name in img_names]
		for img in img_paths:
			for suffix in img_suffix:
				if img.endswith(suffix):
					res_file_list.append(img)
		imgs = [ plt.imread(path) for path in res_file_list ] # 把图片读进列表里面
		return imgs

	def get_obj_img_points(self, grid=(9, 6)):
		cal_imgs = self.get_images_by_dir(self.camera_cal_img_path)
		for img in cal_imgs:
			object_point = np.zeros( (grid[0]*grid[1],3),np.float32 )
			object_point[:,:2] = np.mgrid[0 : grid[0], 0 : grid[1]].T.reshape(-1,2)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			ret, corners = cv2.findChessboardCorners(gray, grid, None)
			if ret:
				self.object_points.append(object_point)
				self.img_points.append(corners)

	def camera_calibrate(self, img):
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.object_points, self.img_points, img.shape[1::-1], None, None)
		dst = cv2.undistort(img, mtx, dist, None, mtx)
		return dst

	def get_M_Minv(self):
		self.M = cv2.getPerspectiveTransform(self.Affine_src, self.Affine_dst)
		self.Minv = cv2.getPerspectiveTransform(self.Affine_dst, self.Affine_src)

	def abs_sobel_thresh(self, img, orient='x', thresh_min=0, thresh_max=255):
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		if orient == 'x':
		    abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
		if orient == 'y':
		    abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
		scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
		binary_output = np.zeros_like(scaled_sobel)
		binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
		return binary_output

	def mag_thresh(self, img, sobel_kernel=3, mag_thresh=(0, 255)):
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
		sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
		gradmag = np.sqrt(sobelx**2 + sobely**2)
		scale_factor = np.max(gradmag)/255 
		gradmag = (gradmag/scale_factor).astype(np.uint8) 
		binary_output = np.zeros_like(gradmag)
		binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
		return binary_output

	def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi/2)):
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
		sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
		absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
		binary_output =  np.zeros_like(absgraddir)
		binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
		return binary_output

	def hls_select(self, img,channel='s',thresh=(0, 255)):
		hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
		if channel=='h':
			channel = hls[:,:,0]
		elif channel=='l':
			channel=hls[:,:,1]
		else:
			channel=hls[:,:,2]
		binary_output = np.zeros_like(channel)
		binary_output[(channel > thresh[0]) & (channel <= thresh[1])] = 1
		return binary_output

	def luv_select(self, img, thresh=(0, 255)):
		luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
		l_channel = luv[:,:,0]
		binary_output = np.zeros_like(l_channel)
		binary_output[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 1
		return binary_output

	def lab_select(self, img, thresh=(0, 255)):
		lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
		b_channel = lab[:,:,2]
		binary_output = np.zeros_like(b_channel)
		binary_output[(b_channel > thresh[0]) & (b_channel <= thresh[1])] = 1
		return binary_output

	def find_line(self, binary_warped):
		histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
		#print(histogram)
		out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
		midpoint = np.int(histogram.shape[0]/2)
		leftx_base = np.argmax(histogram[:midpoint])
		rightx_base = np.argmax(histogram[midpoint:]) + midpoint
		nwindows = 9
		window_height = np.int(binary_warped.shape[0] / nwindows)
		nonzero = binary_warped.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		leftx_current = leftx_base
		rightx_current = rightx_base
		margin = 100
		minpix = 50
		left_lane_inds = []
		right_lane_inds = []

		for window in range(nwindows):
			win_y_low = binary_warped.shape[0] - (window+1)*window_height
			win_y_high = binary_warped.shape[0] - window*window_height
			win_xleft_low = leftx_current - margin
			win_xleft_high = leftx_current + margin
			win_xright_low = rightx_current - margin
			win_xright_high = rightx_current + margin
			good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
			(nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
			good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
			(nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
			left_lane_inds.append(good_left_inds)
			right_lane_inds.append(good_right_inds)
			if len(good_left_inds) > minpix:
				leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
			if len(good_right_inds) > minpix:        
				rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

		left_lane_inds = np.concatenate(left_lane_inds)
		right_lane_inds = np.concatenate(right_lane_inds)
		leftx = nonzerox[left_lane_inds]
		lefty = nonzeroy[left_lane_inds] 
		rightx = nonzerox[right_lane_inds]
		righty = nonzeroy[right_lane_inds] 
		left_fit = np.polyfit(lefty, leftx, 2)
		right_fit = np.polyfit(righty, rightx, 2)
		ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
		left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
		out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
		out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
		# plt.subplot(234)
		# plt.imshow(out_img)
		# plt.axis("off")
		# plt.plot(left_fitx, ploty, color='yellow')
		# plt.plot(right_fitx, ploty, color='yellow')
		# plt.xlim(0, 1280)
		# plt.ylim(720, 0)
		#plt.show()
		return left_fit, right_fit, left_lane_inds, right_lane_inds,out_img,left_fitx,right_fitx,ploty

	def find_line_by_previous(self, binary_warped, left_fit, right_fit):
		nonzero = binary_warped.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		margin = 100
		left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
		left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
		left_fit[1]*nonzeroy + left_fit[2] + margin))) 
		right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
		right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
		right_fit[1]*nonzeroy + right_fit[2] + margin)))  
		leftx = nonzerox[left_lane_inds]
		lefty = nonzeroy[left_lane_inds] 
		rightx = nonzerox[right_lane_inds]
		righty = nonzeroy[right_lane_inds]
		left_fit = np.polyfit(lefty, leftx, 2)
		right_fit = np.polyfit(righty, rightx, 2)
		return left_fit, right_fit, left_lane_inds, right_lane_inds

	def draw_area(self, undist, binary_warped, Minv, left_fit, right_fit):
		ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
		plt.subplot(223)
		plt.imshow(binary_warped, cmap ='gray')
		left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
		warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
		color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
		pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
		pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
		pts = np.hstack((pts_left, pts_right))
		cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
		newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
		result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
		return newwarp, result

	def calculate_curv_and_pos(self, binary_warped, left_fit, right_fit):
		ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
		leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
		ym_per_pix = 30 / 720
		xm_per_pix = 3.7 / 700
		y_eval = np.max(ploty)
		left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
		right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
		left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
		right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
		curvature = ((left_curverad + right_curverad) / 2)
		lane_width = np.absolute(leftx[719] - rightx[719])
		lane_xm_per_pix = 3.7 / lane_width
		veh_pos = (((leftx[719] + rightx[719]) * lane_xm_per_pix) / 2.)
		cen_pos = ((binary_warped.shape[1] * lane_xm_per_pix) / 2.)
		distance_from_center = cen_pos - veh_pos
		return curvature,distance_from_center

	def select_yellow(self, image):
		hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
		lower = np.array([20,60,60])
		upper = np.array([38,174, 250])
		mask = cv2.inRange(hsv, lower, upper)
		return mask

	def select_white(self, image):
		lower = np.array([170,170,170])
		upper = np.array([255,255,255])
		mask = cv2.inRange(image, lower, upper)
		return mask

	def draw_values(self, img, curvature, distance_from_center):
		font = cv2.FONT_HERSHEY_SIMPLEX
		radius_text = "Radius of Curvature: %sm"%(round(curvature))
		if distance_from_center>0:
			pos_flag = 'right'
		else:
			pos_flag= 'left'
		cv2.putText(img,radius_text,(100,100), font, 1,(255,255,255),2)
		center_text = "Vehicle is %.3fm %s of center"%(abs(distance_from_center),pos_flag)
		cv2.putText(img,center_text,(100,150), font, 1,(255,255,255),2)
		return img