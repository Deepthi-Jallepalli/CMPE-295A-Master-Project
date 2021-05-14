# ================================================================
#
#   File name   : detection_custom.py
#   Description : Performs object detection for images
#
#================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import detect_image, detect_realtime, detect_video, Load_Yolo_model, detect_video_realtime_mp
from yolov3.configs import *


image_path = "/home/014505660/fl_project/kitti_dataset/testing/image_2/000010.png"

yolo = Load_Yolo_model()
detect_image(yolo, image_path, "./IMAGES/test_1_detect.jpg", input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
