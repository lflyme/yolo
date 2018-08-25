# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 17:35:45 2018

@author: Administrator
"""
import os
import argparse
import datetime
import numpy as np
import tensorflow as tf
import yolo.config as cfg
from yolo.yolo_net import YOLONet
from utils.timer import Timer
from utils.pascal_voc import pascal_voc

slim = tf.contrib.slim

class Solver(object):
	def __init__(self,net,data):
		self.net = net
		self.data = data
		self.weights_file = cfg.WIEGHTS_FILE
		self.max_iter = cfg.MAX_ITER
		self.initial_learning_rate = cfg.LEARNING_RATE
		self.decay_steps = cfg.DECAY_RATE
		self.decay_rate = cfg.DECAY_RATE
		self.staircase = cfg.STAIRCASE
		self.save_iter = cfg.SAVE_ITER
		self.output_dir = os.path.join(
			cfg.OUTPUT_DIR,datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
		
		if not os.path.exists(self.output_dir):
			os.makedirs(self.output_dir)
		self.save_cfg()
		
		self.variable_to_restore = tf.global_variables()
		self.saver = tf.train.Saver(self.variable_to_restore,max_to_keep = None)
		








