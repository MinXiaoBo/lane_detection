# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 11:22:29 2017
@author: yang
"""

import os
import cv2
import utils
import matplotlib.pyplot as plt
import numpy as np

#script mainly use for drawing the demo picture

cal_imgs = utils.get_images_by_dir('camera_cal')
object_points,img_points = utils.calibrate(cal_imgs,grid=(9,6))

test_imgs = utils.get_images_by_dir('test_img_one')
# test_imgs = utils.get_images_by_dir('new_test')


trans_on_test=[]
for img in test_imgs:
    src = np.float32([[(200, 720), (585, 470), (695, 470), (1120, 720)]])
    dst = np.float32([[(320, 720), (320, 0), (960, 0), (960, 720)]])
    M = cv2.getPerspectiveTransform(src, dst)
    trans = cv2.warpPerspective(img, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    trans_on_test.append(trans)
    
thresh = []
binary_wrapeds = []
histogram = []
for img in test_imgs:
    x_thresh = utils.abs_sobel_thresh(img, orient='x', thresh_min=55, thresh_max=100)
    mag_thresh = utils.mag_thresh(img, sobel_kernel=3, mag_thresh=(70, 255))
    dir_thresh = utils.dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3))
    s_thresh = utils.hls_select(img,channel='s',thresh=(160, 255))
    s_thresh_2 = utils.hls_select(img,channel='s',thresh=(200, 240))
    white_mask = utils.select_white(img)
    yellow_mask = utils.select_yellow(img)
    combined = np.zeros_like(mag_thresh)
    combined[((x_thresh == 1) | (s_thresh == 1)) | ((mag_thresh == 1) & (dir_thresh == 1))| (white_mask>0)|(s_thresh_2 == 1) ]=1
    src = np.float32([[(200, 720), (585, 470), (695, 470), (1120, 720)]])
    dst = np.float32([[(320, 720), (320, 0), (960, 0), (960, 720)]])
    M = cv2.getPerspectiveTransform(src, dst)
    binary_warped = cv2.warpPerspective(combined, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    hist = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    histogram.append(hist)
    binary_wrapeds.append(binary_warped)
    thresh.append(combined)
    
i=0
for binary_warped in binary_wrapeds:
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # plt.plot(histogram)
    # plt.show()
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    nwindows = 9
    window_height = np.int(binary_warped.shape[0]/nwindows)
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
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2) 
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
    
    # plt.subplot(1,3,1)
    # plt.imshow(trans)
    # plt.axis("off")
    plt.subplot(1,2,1)
    plt.imshow(binary_warped,cmap ='gray')
    plt.axis("off")

    i+=1

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.subplot(1,2,2)
    plt.imshow(out_img)
    plt.axis("off")
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

plt.show()