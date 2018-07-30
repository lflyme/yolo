import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import pickle
import copy
import yolo.config as cfg

class pascal_voc(object):
	def __init__(self,phase,rebuild = False):
		self.devkil_path = os.path.join(cfg.PASCAL_PATH,'VOCdevkit')
		self.data_path = os.path.join(self.devkil,'VOC2007')
		self.cache_path = cfg.CACHE_PATH
		self.batch_size = cfg.BATCH_SIZE
		self.image_size = cfg.IMAGE_SIZE
		self.cell_size = cfg.CELL_SIZE
		self.classes = cfg.CLASSES
		self.class_to_ind = dict(zip(self.classes,range(len(self.classes))))
		self.flipped = cfg.FLIPPED
		self.phase = phase
		self.rebuild = 


