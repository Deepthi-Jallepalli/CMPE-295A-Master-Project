#================================================================
#
#   File name   : server_aggregation_weights.py
#   Description : weights aggregation logic
#
#================================================================
from multiprocessing import Process, Queue, Pipe
import cv2
import time
import random
import colorsys
import numpy as np
import tensorflow as tf
from yolov3.configs import *
from yolov3.yolov4 import *
from tensorflow.python.saved_model import tag_constants
import os

def aggregation():
    print('================= Starting Aggregation ================ \n')
    
    w = []
    
    target_dir = '/home/014505660/fl_project/TensorFlow-2.x-YOLOv3/Clients/'
    for root, dirs, files in os.walk(target_dir, topdown=False):
        for name in files:
            print(os.path.join(root, name))
            if "data" in name:
                checkpoint = os.path.join(root, TRAIN_MODEL_NAME)
                print(checkpoint)
                print("Loading custom weights from:", checkpoint,'\n')
                print ("yolo input size:", YOLO_INPUT_SIZE,'\n')
                yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
                yolo.load_weights(checkpoint)  # use custom weights
                weights1 = yolo.get_weights()
                del yolo
                if weights1:
                    w.append(weights1)

    # clients = [1615, 1725, 1662, 1632]
    # clients = [1615, 1662, 1632]
    clients = [1725,1632]
    total_size = sum(clients)
    new_weights = [np.zeros(param.shape) for param in w[0]]
    for c in range(len(w)):
        for i in range(len(new_weights)):
            
            new_weights[i] += w[c][i] * (clients[c]/total_size
                                )
    current_weights = new_weights
    # print("aggregated weights:",  current_weights[0][0])
    yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
    yolo.set_weights(current_weights)
    yolo.save_weights(f'/home/014505660/fl_project/TensorFlow-2.x-YOLOv3/server_agg_weights/{TRAIN_MODEL_NAME}')