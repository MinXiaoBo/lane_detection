#coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
import cv2

def maxinum(image):
	img_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
	rows,cols,_ = image.shape
	dist = np.zeros((rows,cols),dtype=image.dtype)

	for y in range(rows):
		for x in range(cols):
			avg = sum(image[y,x]) / 3
			dist[y,x] = np.uint8(avg)
	return dist

def average(image):
	img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	rows, cols, _ = image.shape
	dist = np.zeros((rows, cols), dtype = image.dtype)
	for y in range(rows):
		for x in range(cols):
			avg = max(image[y, x][0], image[y, x][1], image[y, x][2])
			dist[y, x] = np.uint8(avg)
	return dist

def Weighted(image):
	img_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
	rows,cols,_ = image.shape
	dist = np.zeros((rows,cols),dtype=image.dtype)
    
	for y in range(rows):
		for x in range(cols):
			r,g,b = img_rgb[y,x]
			r = np.uint8(r * 0.299)
			g = np.uint8(g * 0.587)
			b = np.uint8(b * 0.114)
			rgb = np.uint8(r * 0.299 + b * 0.114 + g * 0.587)
			dist[y,x] = rgb
	return dist

def convert_grey(ori_img):
	res_maxinum = maxinum(ori_img)
	res_average = average(ori_img)
	res_Weighted = Weighted(ori_img)
	cv2.imshow('original', ori_img)
	cv2.imshow('maxinum', res_maxinum)
	cv2.imshow('average', res_average)
	cv2.imshow('Weighted', res_Weighted)

def blur(ori_img):
	grey_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2GRAY)
	blur = cv2.blur(grey_img, (5, 5)) #均值滤波
	medianblur = cv2.medianBlur(grey_img, 5) # 中值滤波
	gaussianblur = cv2.GaussianBlur(grey_img, (5,5),0) # 高斯滤波
	cv2.imshow('grey_img', grey_img)
	cv2.imshow('blur', blur)
	cv2.imshow('medianblur', medianblur)
	cv2.imshow('gaussianblur', gaussianblur)

def edge_detection(ori_img):
	grey_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2GRAY)
	gaussianblur = cv2.GaussianBlur(grey_img, (5,5),0)
	img_sobel_x = cv2.Sobel(grey_img, cv2.CV_64F, 1, 0, ksize=3)
	img_sobel_y = cv2.Sobel(grey_img, cv2.CV_64F, 0, 1, ksize=3)
	absX = cv2.convertScaleAbs(img_sobel_x)   # 转回uint8
	absY = cv2.convertScaleAbs(img_sobel_y)
	img_sobel = cv2.addWeighted(absX,0.5,absY,0.5,0)
	img_laplace = cv2.Laplacian(grey_img, cv2.CV_64F, ksize=3)
	img_canny = cv2.Canny(grey_img, 100 , 150)
	cv2.imshow('ori_img', ori_img)
	cv2.imshow('Sobel', img_sobel)
	cv2.imshow('Laplace', img_laplace)
	cv2.imshow('Canny', img_canny)

def Canny(ori_img):
	cv2.imshow('ori_img', ori_img)
	grey_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2GRAY)
	gaussianblur = cv2.GaussianBlur(grey_img, (3,3),0)
	img_canny = cv2.Canny(gaussianblur, 100 , 200)
	cv2.imshow('Canny', img_canny)

src = cv2.imread('./test_img/1.jpg')
# convert_grey(src)
# blur(src)
# edge_detection(src)
Canny(src)
cv2.waitKey()
cv2.destroyAllWindows()