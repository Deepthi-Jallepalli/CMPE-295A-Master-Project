import os
from xml.etree.ElementTree import dump
import json
import pprint

import argparse

from Format import KITTI,YOLO

def main(config):

    if config["datasets"] == "KITTI":
        kitti = KITTI()
        yolo = YOLO(config["cls_list"])

        flag, data = kitti.parse(
            config["label"], config["img_path"], img_type=config["img_type"])

        if flag == True:
            flag, data = yolo.generate(data)

            if flag == True:
                flag, data = yolo.save(data, config["output_path"], config["img_path"],
                                       config["img_type"], config["manifest_path"])

                if flag == False:
                    print("Saving Result : {}, msg : {}".format(flag, data))

            else:
                print("YOLO Generating Result : {}, msg : {}".format(flag, data))

        else:
            print("KITTI Parsing Result : {}, msg : {}".format(flag, data))

    else:
        print("Unkwon Datasets")


if __name__ == '__main__':

    config = {
        "datasets": "KITTI",
        "img_path": "D:\\convert2Yolo-master\\convert2Yolo-master\\custom_data\\kitti\\images",
        "label": "convert2Yolo-master\custom_data\kitti\labels",
        "img_type": ".png",
        "manifest_path": "convert2Yolo-master",
        "output_path": "convert2Yolo-master",
        "cls_list": "D:\\convert2Yolo-master\\convert2Yolo-master\\custom_data\\kitti\\names.txt",
    }

    main(config)

