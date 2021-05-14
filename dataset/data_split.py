#================================================================
#
#   File name   : data_split.py
#   Description : Data split for four clients
#
#================================================================
import sys
import re
import os
# pattern = re.compile("<(\d{4,5})>")

f = open('/home/014352130/fl_project/kitti_train_backup.txt')

def move(line, num):

        getClass = {'0': 'DontCare', '1': 'Misc', '2': 'Person_sitting',
                    '3': 'Cyclist', '4': 'Car', '5': 'Pedestrian', '6': 'Truck',
                    '7': 'Van', '8': 'Tram'}
        classFile = f'./{getClass[num]}.txt'
        fp = open(classFile, "a")
        fp.write(line)
        fp.close()

for line in f.readlines():

        bb = line.split(" ", 1)[1]

        bb_list = bb.split(" ")
        bb_list = [each.split(",")for each in bb_list]
        #print (bb_list)
        #break
        for each in bb_list:
            if "3" in each[-1].strip():
                move(line, "3")
            if "2" in each[-1].strip():
                move(line, "2")
            if "5" in each[-1].strip():
                move(line, "5")
            elif "6" in each[-1].strip():
                move(line, "6")
            elif "7" in each[-1].strip():
                move(line, "7")
            elif "8" in each[-1].strip():
                move(line, "8")
            elif "4" in each[-1].strip():
                move(line, "4")
            elif "0" in each[-1].strip():
                move(line, "0")
            else:
                move(line, "1")
            break

f.close()