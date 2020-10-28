# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 12:47:01 2019

@author: Keshik
"""

from torch.utils.data.dataset import Dataset
import os
import numpy as np
from random import shuffle
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import resize
import torch
import imgaug as ia
from imgaug import augmenters as iaa
import random
import glob

class KITTI2D(Dataset):
    """
        Pytorch Dataset class for KITTI-2D <http://www.cvlibs.net/datasets/kitti/> Object Detection Dataset
        
        Args
            image_dir (string): Directory holding the KITTI-2D Dataset Images
            label_dir (string): Directory holding the correspoding labels
            image_size (tuple): Integer tuple defining the size of input image (Default: 416x416)
            max_objects (int): Maximum cap on objects present on the scene.
            fraction (float): 0 < fraction <= 1.0 defining the subset of images to use. (Primarily for debugging purposes)
            split_ratio (float): 0 < split_ratio <= 1.0 defining the amount of split for training and validation (default: 80% train/ 20% validation)    
    
    """
    def __init__(self, image_dir, 
                 label_dir,
                 image_size = (416, 416),
                 max_objects = 50,
                 fraction = 1.0, 
                 split_ratio=0.8, 
                 train=True):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.fraction = fraction
        self.split_ratio = split_ratio
        self.train = train
        self.image_filenames = []
        self.max_objects = max_objects
        self.state_variables = {"w":0, "h":0, "pad": (0, 0), "padded_h":0, "padded_w":0 }
        self._load_filenames()
        
    
    
    def __getitem__(self, index):
        """
            Args
                index (int): Index
    
            Returns
                tuple: (image_paths, image, labels) where labels is a yolo vector of [max_objects x 5]
            
        """
        # Returns img_path, img(as PIL), bbox (as np array), labels (as np array)
        image = self._read_image(index)
        label = self._read_label(index)
            
        return self._get_img_path(index), image, label
    
    
    def __len__(self):
        """
            Returns
                size of the dataset
        """
        return len(self.image_filenames)
    
    
    def _get_img_path(self, index):
        """
            Args
                index (int): Index
            
            Returns
                relative path of image
        """
        return "{}/{}".format(self.image_dir, self.image_filenames[index])
    
    
    def _read_image(self, index):
        """
            Read the index and return the correponding image as torch tensor with channels first.
            Image augmentations are applied if self.train = True
            
            Args
                index (int): Index
                augment_image (bool): If true, augment the image for variability. (Set to False during validation/ test)
            
            Returns
                Image tensor with channels first
        
        """
        # read the file and return the label
        img_path = self._get_img_path(index)
        image = Image.open(img_path)
        
        # Convert grayscale images to rgb
        if (image.mode != "RGB"):
            image = image.convert(mode = "RGB")            
        
        if self.train:
            augmented_image = self._augment_image(np.asarray(image))
            return torch.from_numpy(self._pad_resize_image(augmented_image, self.image_size))
        
        return torch.from_numpy(self._pad_resize_image(np.asarray(image), self.image_size))
    
    
    def _pad_resize_image(self, image, image_size):
        """
            Pad and resize the image maintaining the aspect ratio of the image
            
            Args
                image (np array): 3-dimensional np image array 
                image_size (tuple): Integer tuple indicating the size of the image
            
            Returns
                Padded and resized image as numpy array with channels first
        
        """
        h, w, _ = image.shape
        dim_diff = np.abs(h - w)
        
        # Upper left padding
        pad1 = dim_diff//2
        
        # lower right padding
        pad2 = 0
        
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        
        new_img = np.pad(image, pad, 'constant', constant_values = 128)/255.0
        padded_h, padded_w, _ = new_img.shape
        
        new_img = resize(new_img, (*image_size, 3), mode='reflect')
        
        # Channels first for torch operations
        new_img = np.transpose(new_img, (2, 0, 1))
        
        # modify state variables
        self.state_variables["h"] = h
        self.state_variables["w"] = w
        self.state_variables["pad"] = pad
        self.state_variables["padded_h"] = padded_h
        self.state_variables["padded_w"] = padded_w
        
        return new_img
        
    
    
    def _read_label(self, index):
        """
            Read the txt file corresponding to the label and output the label tensor following the yolo format [max_objects x 5]
            
            Args
                index (int): Index
            
            Returns
                Torch tensor that encodes the labels for the image
        
        """
        image_filename = self.image_filenames[index]
        label_filename = self._get_label_filename(image_filename)
        
        labels = None
        
        if os.path.exists(label_filename):
            labels = np.loadtxt(label_filename).reshape(-1, 5)
            # Access state variables
            w, h, pad, padded_h, padded_w = self.state_variables["w"], self.state_variables["h"], self.state_variables["pad"], self.state_variables["padded_h"], self.state_variables["padded_w"]
            
            # Extract coordinates for unpadded + unscaled image
            x1 = w * (labels[:, 1] - labels[:, 3]/2)
            y1 = h * (labels[:, 2] - labels[:, 4]/2)
            x2 = w * (labels[:, 1] + labels[:, 3]/2)
            y2 = h * (labels[:, 2] + labels[:, 4]/2)
            
            # Adjust for added padding
            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][0]
            y2 += pad[0][0]
            
            # Calculate ratios from coordinates
            labels[:, 1] = ((x1 + x2) / 2) / padded_w
            labels[:, 2] = ((y1 + y2) / 2) / padded_h
            labels[:, 3] *= w / padded_w
            labels[:, 4] *= h / padded_h
            
        
        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))
        
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        
        filled_labels = torch.from_numpy(filled_labels)

        return filled_labels
            
    
    def _load_filenames(self):
        """
            Load filenames to the class variables in order to be processed later
        
        """
        # Load filenames wth absolute paths
        #self.img_filenames = ["{}/{}".format(self.image_dir, i) for i in os.listdir(self.image_dir)]
        self.image_filenames = os.listdir(self.image_dir)
        self.image_filenames.sort()
        
        # Shuffle and sample dataset
        #shuffle(self.image_filenames)
        self.image_filenames = self.image_filenames[:int(len(self.image_filenames)*self.fraction)]
        
        # Create splitted dataset
        if self.train:
            self.image_filenames = self.image_filenames[:int(len(self.image_filenames)*self.split_ratio)]
        else:
            self.image_filenames = self.image_filenames[int(len(self.image_filenames)*self.split_ratio):]
        

    def _get_label_filename(self, image_filename):
        """
            Given the image filename, get the relative path of the corresponding label file
            
            Args
                image_filename (string): filename i.e. 000000.png
            
            Returns
                label_file path (string): i.e. ./data/train/labels/00000.txt
        """
        # eg:00000.png
        image_id = image_filename.split(".")[0]
        return "{}/{}.txt".format(self.label_dir, image_id)
    
    
    def _augment_image(self, image):
        """
            Augment a single image.
            Uses https://github.com/aleju/imgaug library
            Included in requirements (Do not attempt to manually install, use pip install - requirements.txt)
            
            Args
                image (np array): 3-dimensional np image array
            
            Returns
                Augmented image as 3-dimensional np image array
        """
        # Add label noise
        rand_int = random.randint(-5 , 5)
        value = 0 if rand_int < 0 else rand_int
        
        seq = iaa.Sequential([
            iaa.SomeOf((0, 2)),
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 0.75)), # emboss images
                iaa.OneOf([
                    iaa.GaussianBlur((0, 2.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(5, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
            
                iaa.OneOf([
                    # either change the brightness of the whole image (sometimes
                    # per channel) or change the brightness of subareas
                    iaa.Multiply((0.8, 1.2), per_channel=0.5),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                ]),
            
                iaa.OneOf([
                    iaa.Dropout(p=0.05, per_channel=True),
                    iaa.Crop(px=(0, value)), # crop images from each side by 0 to 4px (randomly chosen)
                ])
        ])
    
        return seq.augment_image(np.asarray(image))



class ImageFolder(Dataset):
    """
        Pytorch Dataset class for KITTI-2D Image evaluation
        
        Args
            folder_path (string): Directory holding the KITTI-2D Dataset Evaluation images
            img_size (int): Size of the image

    """
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))
        self.img_shape = (img_size, img_size)

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image
        img = np.array(Image.open(img_path))
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        return img_path, input_img

    def __len__(self):
        return len(self.files)





        
        
