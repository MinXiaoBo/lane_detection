#!/usr/local/bin/python3

import cv2
import os


def save_img():
    video_path = './video_input/project_video.mp4'
    vc = cv2.VideoCapture(video_path) #读入视频文件
    count = 0
    rval = vc.isOpened()
    while rval:   # 循环读取视频帧
        count = count + 1
        rval, frame = vc.read()
        pic_path = './new_test_img/'
        if rval:
            if count == 1:
                cv2.imwrite(pic_path + 'test_' + str(count) + '.jpg', frame)
            elif count == 100:
                cv2.imwrite(pic_path + 'test_' + str(count) + '.jpg', frame)
            elif count == 200:
                cv2.imwrite(pic_path + 'test_' + str(count) + '.jpg', frame)
            elif count == 300:
                cv2.imwrite(pic_path + 'test_' + str(count) + '.jpg', frame)
            elif count == 400:
                cv2.imwrite(pic_path + 'test_' + str(count) + '.jpg', frame)
            elif count == 500:
                cv2.imwrite(pic_path + 'test_' + str(count) + '.jpg', frame)
        else:
            break

    vc.release()
    print('save_success')
save_img()