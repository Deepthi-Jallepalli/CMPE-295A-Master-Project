# KITTI 2D Object Detection
- [KITTI 2D Object Detection](#kitti-2d-object-detection)
  * [Problem Statement](#problem-statement)
  * [Data](#data)
  * [Data Augmentations](#data-augmentations)
  * [Data splits](#data-splits)
  * [Evaluation Metrics](#evaluation-metrics)
  * [Neural Network Architecture](#neural-network-architecture)
  * [Implementation](#implementation)
  * [Results](#results)
  * [Test on Samples](#test-on-samples)
  * [How to reproduce the code](#how-to-reproduce-the-code)
  * [References](#references)
  
## Problem Statement
The goal of this project is to detect object from a number of visual object classes in realistic scenes. There are 7 object classes:
- Car, Van, Truck, Tram
- Pedestrian, Person
- Cyclist

## Data
The training and test data are ~6GB each (12GB in total). The data can be downloaded at http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark .The label data provided in the KITTI dataset corresponding to a particular image includes the following fields. The labels also include 3D data which is out of scope for this project.


| Key       	| Values 	| Description                                                                                                           	|
|-----------	|--------	|-----------------------------------------------------------------------------------------------------------------------	|
| type      	| 1      	| String describing the type of object: [Car, Van, Truck, Pedestrian,Person_sitting, Cyclist, Tram, Misc or DontCare]   	|
| truncated 	| 1      	| Float from 0 (non-truncated) to 1 (truncated), where truncated refers to the object leaving image boundaries          	|
| occluded  	| 1      	| Integer (0,1,2,3) indicating occlusion state:  0 = fully visible 1 = partly occluded 2 = largely occluded 3 = unknown 	|
| alpha     	| 1      	| Observation angle of object ranging from [-pi, pi]                                                                    	|
| bbox      	| 4      	| 2D bounding box of object in the image (0-based index): contains left, top, right, bottom pixel coordinates           	|

## Data Augmentations
Since the only has 7481 labelled images, it is essential to incorporate data augmentations to create more variability in available data. The following list provides the types of image augmentations performed.
- Image Embossing
- Blur (Gaussian, Average, Median)
- Brightness variation with per-channel probability
- Adding Gaussian Noise with per-channel probability
- Random dropout of pixels

Geometric augmentations are thus hard to perform since it requires modification of every bounding box coordinate and results in changing the aspect ratio of images. We plan to implement Geometric augmentations in the next release. Examples of image embossing, brightness/ color jitter and Dropout are shown below.

![alt text](./readme_resources/augmentations_final.png)

**Adding Label Noise**  
To allow adding noise to our labels to make the model robust, We performed side by side of cropping images where the number of pixels were chosen from a uniform distribution of [-5px, 5px] where values less than 0 correspond to no crop.

## Data splits
We used an 80 / 20 split for train and validation sets respectively since a separate test set is provided.

## Evaluation Metrics
We use mean average precision (mAP) as the performance metric here.  
**Average Precision:** It is the average precision over multiple IoU values.  
**mAP:** It is average of AP over all the object categories.  

## Neural Network Architecture
We experimented with faster R-CNN, SSD (single shot detector) and YOLO networks. We chose YOLO V3 as the network architecture for the following reasons,
1. YOLO V3 is relatively lightweight compared to both SSD and faster R-CNN, allowing me to iterate faster.
2. Costs associated with GPUs encouraged me to stick to YOLO V3.
3. We wanted to evaluate performance real-time, which requires very fast inference time and hence we chose YOLO V3 architecture.

## Implementation
We implemented YoloV3 with Darknet backbone using Pytorch deep learning framework.
![alt text](./readme_resources/yolov3.png)

## Results
![alt text](./readme_resources/exp1-loss-final.JPG)
![alt text](./readme_resources/exp3-ap.JPG)

## Test on samples
Use the detect.py script to test the model on sample images at /data/samples. Feel free to put your own test images here. The results are saved in /output directory. Some inference results are shown below.
![alt text](./readme_resources/detection_collage.jpg)

## How to reproduce the code
1. Install dependencies : pip install -r requirements.txt
2. Directory structure
    * /src: contains source code
    * /data: data directory for KITTI 2D dataset 
      - samples/
      - train/
        - images/ (Place all training images here)
        - yolo_labels/ (This is included in the repo)
      - test/
        - images/ (Place all test images here)
      - names.txt (Contains the object categories)
      - readme.txt (Official KITTI Data Documentation)
    * /config: contains yolo configuration file
    * /readme_resources: 
3. Run the main function in main.py with required arguments. The codebase is clearly documented with clear details on how to execute the functions. You need to interface only with this function to reproduce the code.
4. Pre-trained weights can be obtained at https://drive.google.com/open?id=1qvv5j59Vx3rg9GZCYW1WwlvQxWg4aPlL

## References
- https://github.com/eriklindernoren/PyTorch-YOLOv3
- https://github.com/ssaru/convert2Yolo
- https://github.com/BobLiu20/YOLOv3_PyTorch
- https://github.com/packyan/PyTorch-YOLOv3-kitti
