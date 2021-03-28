# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 15:17:27 2021

@author: samgao1999
"""

import cv2
import paddlehub as hub
import os
import numpy as np
from PIL import Image

'''
#############
需要注意的一点是：摄像头的分辨率不方便轻易调整，请将图腾动画视频的分辨率与摄像头分辨率保持一致
#############
'''
class StandMerge:
    
    output_path = os.path.join(os.path.dirname(__file__), "video_out_put", "1.avi")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    def __init__(self, img, realtime, animation):
        self.img = img
        
        self.animation = animation
        
        self.realtime = realtime
        
        self.cap_out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 
                                            24, (1024, 728))  # 设置输出的分辨率和帧率
               
        self.human_seg = hub.Module(name='humanseg_server')
        
    
    def set_output_path(self, path):
        self.output_path = path
        
        
    def video_init(self):
        img = self.img
        realtime = self.realtime
        animation = self.animation
        cap_out = self.cap_out
        
        # self.animation = cv2.VideoCapture(r"C:\Users\samgao1999\Desktop\tatoo\test\star.mp4")  # 读取视频的路径
        animation.set(cv2.CAP_PROP_FRAME_HEIGHT, int(animation.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # 设置图腾动画部分分辨率
        
        animation.set(cv2.CAP_PROP_FRAME_WIDTH, int(animation.get(cv2.CAP_PROP_FRAME_WIDTH)))  
        
        animation.set(cv2.CAP_PROP_FPS, 24)  # 设置图腾动画部分帧率
        
        # realtime = cv2.VideoCapture(0)
        if (realtime == None):
            print("none")
            pass
        else:
            realtime.set(cv2.CAP_PROP_FRAME_HEIGHT, int(realtime.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # 设置实时视频部分分辨率
            
            realtime.set(cv2.CAP_PROP_FRAME_WIDTH, int(realtime.get(cv2.CAP_PROP_FRAME_WIDTH)))  
            
            realtime.set(cv2.CAP_PROP_FPS, 24)  # 设置实时视频部分帧率
            
            width = max(int(animation.get(cv2.CAP_PROP_FRAME_WIDTH)), int(realtime.get(cv2.CAP_PROP_FRAME_WIDTH)))
            
            height = max(int(animation.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(realtime.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            
            fps = 24
            
            cap_out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 
                                      fps, (width, height))  # 设置输出的分辨率和帧率
        
        if (img.any() == None):
            print("none")
            pass
        else:
            width = max(int(animation.get(cv2.CAP_PROP_FRAME_WIDTH)), img.shape[1])
            
            height = max(int(animation.get(cv2.CAP_PROP_FRAME_HEIGHT)), img.shape[0])
            
            fps = 24
            
            size = (width, height)
                        
            cap_out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 
                                      fps, size)  # 设置输出的分辨率和帧率
            print(cap_out)
        return cap_out


    def configuation(self):
        print("\nConfiguation : ")
        print("aniamtion : {}x{}".format(self.animation.get(cv2.CAP_PROP_FRAME_WIDTH), 
                                         self.animation.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        
        if (self.realtime == None):
            print("realtime : {}".format(str(None)))
        else:
            print("realtime : {}x{}".format(self.realtime.get(cv2.CAP_PROP_FRAME_WIDTH), 
                                            self.realtime.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            
        if (self.img.any() == None):
            print("img : {}".format(str(None)))
        else:
            print("img : {}x{}".format(self.img.shape[1], self.img.shape[0]))
            
        print("output : {}x{}".format(self.cap_out.get(cv2.CAP_PROP_FRAME_WIDTH), 
                                      self.cap_out.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    
    def video_subtract_char(self, frame_org, frame_id, bg_im):
        '''
        # frame_org : Array of uint8 图腾动画中获取的图片
        # frame_id : int 图片的id
        # bg_im : Array of uint8 从摄像头拍摄的视频中获取的图片
        
        # 输出的为融合后的图像的一帧
        # 通过调用paddle的human_seg库实现
        '''
        prev_gray = None  # 前一帧图像的灰度图
        prev_cfd = None  # 前一帧光流追踪图和预测结果融合图
        [img_matting, prev_gray, prev_cfd] = self.human_seg.video_stream_segment(frame_org=frame_org,  # 图像截取
                                                                            frame_id=frame_id,
                                                                            prev_gray=prev_gray,
                                                                            prev_cfd=prev_cfd,
                                                                            use_gpu=True)
        
        img_matting = np.repeat(img_matting[:, :, np.newaxis], 3, axis=2)  # 依照格式生成图片矩阵
        bias_top = int(0.5 * (abs(bg_im.shape[0] - img_matting.shape[0])))
        bias_bottom = abs(bg_im.shape[0] - img_matting.shape[0] - bias_top)
        bias_left = int(0.5 * (abs(bg_im.shape[1] - img_matting.shape[1])))
        bias_right = abs(bg_im.shape[1] - img_matting.shape[1]) - bias_left
        frame_org = cv2.copyMakeBorder(frame_org, bias_top, bias_bottom, bias_left, bias_right, cv2.BORDER_CONSTANT, value=0)
        img_matting = cv2.copyMakeBorder(img_matting, bias_top, bias_bottom, bias_left, bias_right, cv2.BORDER_CONSTANT, value=0)
        bg_im = (1-img_matting) * bg_im  # 将摄像头中图片的图腾部分扣去，这片区域的像素变为0
        comb = (img_matting * frame_org + bg_im).astype(np.uint8)  #融合
        
        return comb
    
    
    def video_merge(self, realtime, animation):
        '''
        # realtime : VideoCapture, 摄像头的实时视频 
        # animation : VideoCapture, 图腾动画的视频
        
        # 通过输入实时视频和动画视频，融合图腾和实物图，生成视频并输出
        '''
        
        realtime = self.realtime
        animation = self.animation
        cap_out = self.cap_out
        
        cap_out = self.video_init()
        self.configuation()
        
        
        while (realtime.isOpened() and animation.isOpened()):  # 判断是否播放完视频
            ret_realtime, frame_realtime = realtime.read()
            ret_animation, frame_animation = animation.read()
            if (ret_realtime and ret_animation):
                res = self.video_subtract_char(frame_animation, animation.get(1), frame_realtime)  # 进行图像融合
                cv2.imshow("output",res)  # 显示图片
                cv2.waitKey(25)  # 调整视频快慢
                cap_out.write(res)  # 输入
            else:
                break
            
        animation.release()
        realtime.release()
        cap_out.release()
        cv2.destroyAllWindows() 
        
        
    def img_merge(self, img, animation):
        img = self.img
        animation = self.animation
        cap_out = self.cap_out
                        
        cap_out = self.video_init()
        self.configuation()
        
        while (animation.isOpened()):  # 判断是否播放完视频
            ret_animation, frame_animation = animation.read()
            if (ret_animation):
                res = self.video_subtract_char(frame_animation, animation.get(1), img)  # 进行图像融合
                cv2.imshow("",res)  # 显示图片
                cap_out.write(res)  # 输入
                cv2.waitKey(25)  # 调整视频快慢
            else:
                break
        
        animation.release()
        cap_out.release()
        cv2.destroyAllWindows() 
        
        
if __name__ == "__main__":
    img = cv2.imread(r"C:\Users\samgao1999\Desktop\tatoo\bg_images\bg3.jpg")
    animation = cv2.VideoCapture(r"C:\Users\samgao1999\Desktop\tatoo\stand_video\star1.mp4")
    stand_merge = StandMerge(img, None, animation)
    stand_merge.set_output_path(r"C:\Users\samgao1999\Desktop\tatoo\output\star1.avi")
    stand_merge.img_merge(img, animation)
        
    
