# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 18:51:53 2019

@author: Keshik
"""


from imgaug import augmenters as iaa
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

seq = iaa.Sequential([
    #iaa.Emboss(alpha=(0, 1.0), strength=(2.5, 3)) # emboss images
    
    #iaa.Crop(px=(30, 40)), # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Dropout(p=0.25, per_channel=True),
    #iaa.GaussianBlur((1.0, 2.0)),
    #iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
    #iaa.MedianBlur(k=(3, 11)),
#    iaa.OneOf([
#        iaa.GaussianBlur((0, 2.0)), # blur images with a sigma between 0 and 3.0
#        iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
#        iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
#    ]),
#    
#    
    #iaa.Multiply((0.8, 1.2), per_channel=0.5),
#    iaa.OneOf([
#        # either change the brightness of the whole image (sometimes
#        # per channel) or change the brightness of subareas
#        iaa.Multiply((0.8, 1.2), per_channel=0.5),
#        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
#    ]),
#    
#    iaa.OneOf([
#        iaa.Dropout(p=0.05, per_channel=True),
#        iaa.Crop(px=(0, 4)), # crop images from each side by 0 to 16px (randomly chosen)
#    ])
])


image = Image.open("/content/gdrive/My Drive/KITTI-2d-object-detection-master/KITTI-2d-object-detection-master/data/test/001793.png")

#print(np.asarray(image).shape)
images_aug = seq.augment_image(np.asarray(image))
#print(images_aug.shape)
#plt.imshow(images_aug)
#plt.savefig('emboss-68.png', dpi=1000, bbox_inches='tight',transparent=True, pad_inches=0)
#plt.show()



fig = plt.figure(frameon=False)
fig.set_size_inches(9,3)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(images_aug)
fig.savefig("car.png", dpi=1000)

fig.show()

