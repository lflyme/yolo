# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 22:31:05 2018

@author: Administrator
"""
import os

DATA_PATH = 'data'

PASCAL_PATH = os.path.join(DATA_PATH,'pascal_voc')

CACHE_PATH = os.path.join(PASCAL_PATH,'cache')

OUTPUT_DIR = os.path.join(PASCAL_PATH,'output')

WEIGHTS_DIR = os.path.join(PASCAL_PATH,'weights')

WEIGHTS_FILE = None


CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']


FLIPPED = True



#model parameter
IMAGE_SIZE = 448  #输入图片设置大小

CELL_SIZE = 7

BOXES_PER_CELL = 2

ALPHA = 0.1

DISP_CONSOLE = False

OBJECT_SCALE = 1.0 #目标物体的代价因子
NOOBJECT_SCALE= 1.0 #非目标物体的代价因子
CLASS_SCALE = 2.0 #类别的代价因子
COORD_SCALE = 5.0 #坐标的代价因子










