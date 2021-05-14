# Federated Learning for Object Detection in Autonomous Vehicles

----------------------------------------------------------------
### Table of Contents

* [Team Members](#team-members)
* [Objective](#objective)
* [Dataset](#dataset)
* [Data Conversion to YOLOv3 Format](data-conversion-to-yolov3-format)
* [Data Split](#data-split)
* [Evaluation Metrics](#evaluation-metrics)
* [Implementation](#implementation)
* [Results](#results)
* [Steps to Reproduce](#steps-to-reproduce)
* [References](#references)

## <a name="team-members"></a>Team Members
* Deepthi Jallepalli
* Navya Chennagiri Ravikumar
* Poojitha Vurtur Badarinath
* Shravya Uchil

## Objective
Object detection is extensively used in today's world of AI. It is one of the key features of Autonomous Driving Systems. The current object detection models are highly dependent on centralized data to train. The raising concern with this is data privacy. The data privacy issue can be addressed with the Federated Learning(FL) model proposed in this paper. FL architecture aims at preserving data privacy and improving performance by training the model with decentralized data. Object detection models are trained locally at each node on their proprietary dataset, and the resultant weights are securely aggregated at the global server to yield an improved model. Further, we draw the comparison of object detection performance for an FL versus traditional deep learning approach.

## Dataset
The dataset used in this project is KITTI Dataset, a real-world computer vision benchmark designed specifically for autonomous driving.  
The dataset consists of 7481 training images and 7518 test images. It consists of 8 classes: Car, Van, Truck, Pedestrian, Person sitting, Cyclist, Tram, and Misc. Additionally, few objects are not labelled, perhaps they were too far from the scanner. Hence, they are classified as DontCare.

## Data Conversion to YOLOv3 Format
To run YOLO model with Darknet weights, KITTI labels need to be converted to YOLO format. A text file is generated which contains ground truth of the image in the following format:  
&lt;object-class&gt; &lt;x&&gt; &lt;y&gt; &lt;width&gt; &lt;height&gt;  
where x, y, width, and height are relative to the image's width and height.  
&lt;object-class&gt; has the image path prepended to it.
  
 ## Data Split
 Dataset is split into eight categories (DontCare excluded), ensuring that each image in a category contains at least one object belonging to that class. Further, these categories are distributed among four participating clients such that each client is deficit of a minimum of one class. Since KITTI dataset is imbalanced, in that 'Car' and 'Pedestrian' classes have more occurrences in comparison to other classes, all clients may perhaps have samples belonging to 'Car' and 'Pedestrian'.
 
 ## Evaluation Metrics
 Mean Average Precision (mAP) and Intersection over Union (IoU) are used as the evaluation metrics in this project.
 Intersection over Union is a ratio between the intersection and the union of the predicted boxes and the ground truth boxes.
 mAP compares the ground-truth bounding box to the detected box and returns a score. The higher the score, the more accurate the model is in its detections.
 
 ## Implementation
 - We implemented Object detection using YOLO framework in TensorFlow 
 - The Federated Learning set up incorporates a client-server model
 - We have implemented weights aggregation using Federated Averaging Algorithm
 - We are using connection-oriented TCP protocol to establish communication between the server and the clients. This protocol is implemented using Socket library available in Python.
 - The weights file and client metadata are encrypted in the client node before transferring them via socket to the server. The server decrypts the files, performs secure aggregation of the weights, and encrypts them back before re sending to the client. We have implemented this using Python package called cryptography.

## Results

#### We ran the Object Detection code individually for four clients over 5 epochs, repeated for 15 communication rounds. Below is the results obtained:  


| Client  | Data Size   | Epochs  |Comm Rounds | mAP    |
|-------- |------------ |-------  |----------- | -----  |
| C1      | 1615        | 5   	  | 15         | 68.5%  |
| C2 	    | 1725        | 5       | 1    	     | 66.9%  |
| C3  	  | 1662        | 5       | 15	       | 64.7%  |
| C4      | 1632    	  | 5       | 15         | 64.4%  |

#### Comparison between Deep Learning and Federated Learning Approaches:  

| Training Type                 | Data Size  | Epochs |Total Loss | mAP    |
|-----------------------------  |----------- |------  |---------- |------- |
| Deep Learning                 | 6481       | 5   	  | 21.43     | 44.5%  |
| Federated Learning (3 rounds) |	6481       | 5      | 26.26    	| 46.1%  |
| Deep Learning                 | 6481       | 10     | 16.30	    | 16.3%  |
| Federated Learning (3 rounds) | 6481    	 | 10     | 8.5       | 63%    |

## Steps to Reproduce
- requires TensorFlow with GPU (The code runs without GPU, but slow)
- download the dataset from http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark
- install dependencies and download pretrained weights
 ```
 pip install -r requirements.txt

 # yolov3
 wget -P model_data https://pjreddie.com/media/files/yolov3.weights
 
 ```
- perform data split
 ```
 python dataset/data_split.py
 ```
 - start FL training
  ```
  # start server
  python server_rec_weights.py
  
  # start client
  python client_send_weights.py
  
  ```
  - perform detection
   ```
   python detection_custom.py
   ```
   
   ## References
- https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
