# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 12:47:01 2019

@author: Keshik

Source
    https://github.com/packyan/PyTorch-YOLOv3-kitti
"""

from __future__ import division

from model import Darknet
from utils import non_max_suppression, load_classes

import os
import time
import datetime

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import ImageFolder
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import random

def detect(kitti_weights = '/content/gdrive/My Drive/KITTI-2d-object-detection-master/KITTI-2d-object-detection-master/best_weights_kitti.pth', config_path = '/content/gdrive/My Drive/KITTI-2d-object-detection-master/KITTI-2d-object-detection-master/config/yolov3-kitti.cfg', class_path = '/content/gdrive/My Drive/KITTI-2d-object-detection-master/KITTI-2d-object-detection-master/data/names.txt'):
    """
        Script to run inference on sample images. It will store all the inference results in /output directory (relative to repo root)
        
        Args
            kitti_weights: Path of weights
            config_path: Yolo configuration file path
            class_path: Path of class names txt file
            
    """
    cuda = torch.cuda.is_available()
    os.makedirs('/content/gdrive/My Drive/KITTI-2d-object-detection-master/KITTI-2d-object-detection-master/output/images', exist_ok=True)

    # Set up model
    model = Darknet(config_path, img_size=416)
    model.load_weights(kitti_weights)

    if cuda:
        model.cuda()
        print("Cuda available for inference")

    model.eval() # Set in evaluation mode

    dataloader = DataLoader(ImageFolder("/content/gdrive/My Drive/KITTI-2d-object-detection-master/KITTI-2d-object-detection-master/data/test/", img_size=416),
                            batch_size=2, shuffle=False, num_workers=0)

    classes = load_classes(class_path) # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    imgs = []           # Stores image paths
    img_detections = [] # Stores detections for each image index

    print('data size : %d' % len(dataloader) )
    print ('\nPerforming object detection:')
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, 80, 0.8, 0.4)
            print(detections)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print ('\t+ Batch %d, Inference Time: %s' % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    #cmap = plt.get_cmap('tab20b')
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print ('\nSaving images:')
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print ("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        kitti_img_size = 416
        
        # The amount of padding that was added
        pad_x = max(img.shape[0] - img.shape[1], 0) * (kitti_img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (kitti_img_size / max(img.shape))
        # Image height and width after padding is removed
        unpad_h = kitti_img_size - pad_y
        unpad_w = kitti_img_size - pad_x

        # Draw bounding boxes and labels of detections
        if detections is not None:
            print(type(detections))
            print(detections.size())
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print ('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))
                # Rescale coordinates to original dimensions
                box_h = int(((y2 - y1) / unpad_h) * (img.shape[0]))
                box_w = int(((x2 - x1) / unpad_w) * (img.shape[1]) )
                y1 = int(((y1 - pad_y // 2) / unpad_h) * (img.shape[0]))
                x1 = int(((x1 - pad_x // 2) / unpad_w) * (img.shape[1]))

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                        edgecolor=color,
                                        facecolor='none')
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(x1, y1-30, s=classes[int(cls_pred)]+' '+ str('%.4f'%cls_conf.item()), color='white', verticalalignment='top',
                        bbox={'color': color, 'pad': 0})

        # Save generated image with detections
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.savefig('/content/gdrive/My Drive/KITTI-2d-object-detection-master/KITTI-2d-object-detection-master/output/%d.png' % (img_i), bbox_inches='tight', pad_inches=0.0)
        plt.close()


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    detect()
