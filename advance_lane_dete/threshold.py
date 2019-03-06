#!/usr/local/bin/python3

import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import utils

def get_images_by_dir(dir_name):
    res_file_list = []
    img_suffix = ('.jpg', '.png', '.jpeg')  # 可以使用的图像类型
    if dir_name[-1] == '/':                 # 如果路径结尾有斜杠就把他去掉 如果严谨一点可以判断两次
        dir_name = dir_name[:-1]
    img_names = os.listdir(dir_name)
    img_paths = [dir_name + '/' + img_name for img_name in img_names]
    for img in img_paths:
        for suffix in img_suffix:
            if img.endswith(suffix):
                res_file_list.append(img)
    imgs = [ cv2.imread(path) for path in res_file_list ] # 把图片读进列表里面
    return imgs

def show_img(img_list, flag = 'img_show'):
	plt.figure(flag)
	count = 1
	for image in img_list:
		plt.subplot(len(img_list)/3+1,3,count)
		plt.imshow(image)
		count = count + 1
	plt.show()

def thresholding(img):
	x_thresh = abs_sobel_threshold(img, orient='x', thresh_min=10 ,thresh_max=230)
	plt.subplot(2,4,2)
	plt.imshow(x_thresh, cmap='gray')
	plt.axis("off")
	mag_thresh = mag_threshold(img, sobel_kernel=3, mag_thresh=(30, 150))
	plt.subplot(2,4,3)
	plt.imshow(mag_thresh, cmap='gray')
	plt.axis("off")
	dir_thresh = dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3))
	plt.subplot(2,4,4)
	plt.imshow(dir_thresh, cmap='gray')
	plt.axis("off")
	hls_thresh = hls_select(img, thresh=(180, 255))
	plt.subplot(2,4,5)
	plt.imshow(hls_thresh, cmap='gray')
	plt.axis("off")
	lab_thresh = lab_select(img, thresh=(155, 200))
	plt.subplot(2,4,6)
	plt.imshow(lab_thresh, cmap='gray')
	plt.axis("off")
	luv_thresh = luv_select(img, thresh=(225, 255))
	plt.subplot(2,4,7)
	plt.imshow(luv_thresh, cmap='gray')
	plt.axis("off")
	threshholded = np.zeros_like(mag_thresh)
	threshholded[((x_thresh == 1) & (mag_thresh == 1)) | ((dir_thresh == 1)
	 & (hls_thresh == 1)) | (lab_thresh == 1) | (luv_thresh == 1)] = 1
	return threshholded

def abs_sobel_threshold(img, orient='x', thresh_min=0, thresh_max=255):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	if orient == 'x':
		abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
	if orient == 'y':
		abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
	binary_output = np.zeros_like(scaled_sobel)
	binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
	return binary_output

def mag_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	gradmag = np.sqrt(sobelx**2 + sobely**2)
	scale_factor = np.max(gradmag)/255
	gradmag = (gradmag/scale_factor).astype(np.uint8)
	binary_output = np.zeros_like(gradmag)
	binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
	return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
	binary_output =  np.zeros_like(absgraddir)
	binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
	return binary_output

def hls_select(img,channel='s',thresh=(0, 255)):
	hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
	if channel=='h':
		channel = hls[:,:,0]
	elif channel=='l':
		channel=hls[:,:,1]
	else:
		channel=hls[:,:,2]
	binary_output = np.zeros_like(channel)
	binary_output[(channel > thresh[0]) & (channel <= thresh[1])] = 1
	return binary_output

def luv_select(img, thresh=(0, 255)):
	luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
	l_channel = luv[:,:,0]
	binary_output = np.zeros_like(l_channel)
	binary_output[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 1
	return binary_output

def lab_select(img, thresh=(0, 255)):
	lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
	b_channel = lab[:,:,2]
	binary_output = np.zeros_like(b_channel)
	binary_output[(b_channel > thresh[0]) & (b_channel <= thresh[1])] = 1
	return binary_output

def main():
	img_path = './new_test/'
	ori_img = get_images_by_dir(img_path)
	res_img = []
	for img in ori_img:
		grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		plt.subplot(2,4,1)
		plt.imshow(grey_img, cmap='gray')
		plt.axis("off")
		res = thresholding(img)
		plt.subplot(2,4,8)
		plt.imshow(res, cmap='gray')
		plt.axis("off")
		plt.show()
		res_img.append(res)

	# show_img(res_img)

if __name__ == '__main__':
	main()
