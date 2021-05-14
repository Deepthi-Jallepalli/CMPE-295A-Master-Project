#================================================================
#
#   File name   : voc_to_YOLOv3.py
#   Description : converts dataset from voc to Yolov3 format
#
#================================================================
import xml.etree.ElementTree as ET
from os import getcwd
import os


# dataset_train = 'OID/Dataset/train/'
dataset_train = '/home/014352130/fl_project/kitti_dataset/dataset/training/'
dataset_file = 'kitti_train_trial.txt'
classes_file = dataset_file[:-4]+'_classes.txt'


CLS = os.listdir(dataset_train)
classes =[dataset_train+CLASS for CLASS in CLS]
print("classes: ",classes)
wd = getcwd()
getIndex = {'DontCare': 0, 'Misc': 1, 'Person_sitting': 2,'Cyclist': 3, 'Car': 4, 'Pedestrian': 5, 'Truck':6, 'Van': 7, 'Tram': 8}


def test(fullname):
    bb = ""
    in_file = open(fullname)
    tree=ET.parse(in_file)
    print(tree)
    root = tree.getroot()
    # print("root:", root)
    for i, obj in enumerate(root.iter('object')):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        print("cls: ", cls)
        # if cls not in CLS or int(difficult)==1:
        #     continue
        # cls_id = CLS.index(cls)
        cls_name = obj.find('name').text
        print("class name: ", cls_name)
        cls_id = getIndex[cls_name]
        print("cls_id: ", cls_id)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        bb += (" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        # print(bb)

        # we need this because I don't know overlapping or something like that
        # if cls == 'Traffic_light':
        #     list_file = open(dataset_file, 'a')
        #     file_string = str(fullname)[:-4]+'.png'+bb+'\n'
        #     list_file.write(file_string)
        #     list_file.close()
        #     bb = ""
            

    if bb != "":
        list_file = open(dataset_file, 'a')
        file_string = str(fullname)[:-4]+'.png'+bb+'\n'
        list_file.write(file_string)
        print(file_string)
        list_file.close()
        



for CLASS in classes:
    for filename in os.listdir(CLASS):
        # print("fn: ",filename)
        if not filename.endswith('.xml'):
            continue
        fullname = CLASS+'/'+filename
        print("fullname:", fullname)
        test(fullname)

for CLASS in CLS:
    list_file = open(classes_file, 'a')
    file_string = str(CLASS)+"\n"
    list_file.write(file_string)
    list_file.close()
