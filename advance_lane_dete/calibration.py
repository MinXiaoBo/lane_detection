#!/usr/local/bin/python3
# -*- coding: utf-8 -*-


import os
import cv2
import utils
import matplotlib.pyplot as plt
import numpy as np

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
		plt.subplot(len(img_list)/3,3,count)
		plt.imshow(image)
		count = count + 1
	plt.show()

#function take the chess board image and return the object points and image points
def calibrate(images,grid=(9,6)):
    object_points=[]
    img_points = []
    for img in images:
        object_point = np.zeros( (grid[0]*grid[1],3),np.float32 )
        object_point[:,:2]= np.mgrid[0:grid[0],0:grid[1]].T.reshape(-1,2)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, grid, None)
        if ret:
            object_points.append(object_point)
            img_points.append(corners)
    return object_points,img_points

def cal_undistort(img, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

def main():
    cal_imgs = get_images_by_dir('camera_cal')
    object_points,img_points = calibrate(cal_imgs,grid=(9,6))
    test_imgs = get_images_by_dir('test_img_one')

    undistorted = []
    for img in test_imgs:
        img = cal_undistort(img,object_points,img_points)
        undistorted.append(img)

    for img in test_imgs:
        cv2.imshow('ori_img', img)
    for img in undistorted:
        cv2.imshow('undistorted', img)

    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
