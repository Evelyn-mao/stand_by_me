# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 15:17:27 2021

@author: samgao1999
"""

import cv2
import paddlehub as hub
import os
import numpy as np

'''
#############
需要注意的一点是：摄像头的分辨率不方便轻易调整，请将图腾动画视频的分辨率与摄像头分辨率保持一致
#############
'''

os.environ['CUDA_VISIBLE_DEVICES'] = '0'   # 设置使用GPU

human_seg = hub.Module(name='humanseg_server')  # 调用paddle包

save_path = r"C:\Users\samgao1999\Desktop\tatoo\output\result_3.avi"  #设置存储路径

animation = cv2.VideoCapture(r"C:\Users\samgao1999\Desktop\tatoo\test\star.mp4")  # 读取视频的路径
animation.set(cv2.CAP_PROP_FRAME_HEIGHT, int(animation.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # 设置图腾动画部分分辨率
animation.set(cv2.CAP_PROP_FRAME_WIDTH, int(animation.get(cv2.CAP_PROP_FRAME_WIDTH)))  
animation.set(cv2.CAP_PROP_FPS, 24)  # 设置图腾动画部分帧率


realtime = cv2.VideoCapture(0)  
realtime.set(cv2.CAP_PROP_FRAME_HEIGHT, int(animation.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # 设置实时视频部分分辨率
realtime.set(cv2.CAP_PROP_FRAME_WIDTH, int(animation.get(cv2.CAP_PROP_FRAME_WIDTH)))  
realtime.set(cv2.CAP_PROP_FPS, 24)  # 设置实时视频部分帧率

width = int(animation.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(animation.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 24
cap_out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width, height))  # 设置输出的分辨率和帧率 


def video_subtract_char(frame_org, frame_id, bg_im):
    '''
    # frame_org : Array of uint8 图腾动画中获取的图片
    # frame_id : int 图片的id
    # bg_im : Array of uint8 从摄像头拍摄的视频中获取的图片
    
    # 输出的为融合后的图像的一帧
    # 通过调用paddle的human_seg库实现
    '''
    prev_gray = None  # 前一帧图像的灰度图
    prev_cfd = None  # 前一帧光流追踪图和预测结果融合图
    [img_matting, prev_gray, prev_cfd] = human_seg.video_stream_segment(frame_org=frame_org,  # 图像截取
                                                                        frame_id=frame_id,
                                                                        prev_gray=prev_gray,
                                                                        prev_cfd=prev_cfd,
                                                                        use_gpu=True)
    
    img_matting = np.repeat(img_matting[:, :, np.newaxis], 3, axis=2)  # 依照格式生成图片矩阵
    bg_im = (1-img_matting) * bg_im  # 将摄像头中图片的图腾部分扣去，这片区域的像素变为0
    comb = (img_matting * frame_org + bg_im).astype(np.uint8)  #融合
    
    return comb
 

def merge(realtime, animation):
    '''
    # realtime : VideoCapture, 摄像头的实时视频 
    # animation : VideoCapture, 图腾动画的视频
    
    # 通过输入实时视频和动画视频，融合图腾和实物图，生成视频并输出
    '''
    while (realtime.isOpened() and animation.isOpened()):  # 判断是否播放完视频
        ret_realtime, frame_realtime = realtime.read()
        ret_animation, frame_animation = animation.read()
        if (ret_realtime and ret_animation):
            res = video_subtract_char(frame_animation, realtime.get(1), frame_realtime)  # 进行图像融合
            cv2.imshow("",res)  # 显示图片
            cv2.waitKey(15)  # 调整视频快慢
            cap_out.write(res)  # 输入
        else:
            break
    cv2.waitKey(0)
    animation.release()
    realtime.release()
           

if __name__ == "__main__":
    merge(realtime, animation)