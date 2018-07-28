# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 22:32:26 2018

@author: Administrator
"""
import numpy as np
import tensorflow as tf
import yolo.config as cfg

slim = tf.contrib.slim

class YOLONet(object):
	
	def __init__(self,is_training = True):
		self.classes = cfg.CLASSES
		self.num_class = len(self.classes)
		self.image_size = cfg.IMAGE_SIZE
		self.cell_size = cfg.CELL_SIZE
		self.boxes_per_cell = cfg.BOXES_PER_CELL
		self.output_size = (self.cell_size * self.cell_size) * \
			(self.num_class + self.boxes_per_cell * 5)
			
		self.scale = 1.0 * self.image_size / self.cell_size #缩放比
		self.boundary1 = self.cell_size * self.cell_size * self.num_class #类别的维度边界
		self.boundary2 = self.boundary1 +\
			self.cell_size * self.cell_size * self.boxes_per_cell #box维度的边界
		
		self.object_scale = cfg.OBJECT_SCALE
		self.noobject_scale = cfg.NOOBJECT_SCALE
		self.class_scale = cfg.CLASS_SCALE
		self.coord_scale = cfg.COORD_SCALE
		
		self.learning_rate = cfg.LEARNING_RATE
		self.batch_size = cfg.BATCH_SIZE
		self.alpha = cfg.ALPHA
		
		#[2,7,7] -> [7,7,2]
		self.offset = np.transpose(np.reshape(np.array([np.arange(self.cell_size)]),
						(self.boxes_per_cell,self.cell_size,self.cell_size)),(1,2,0))
			
		#输入变量
		self.images = tf.placeholder(tf.float32,[None,self.image_size,self.image_size,3],
						name = 'images')
		
		#构建网络图
		self.logits = self.build_network(self.images,num_outputs = self.output_size,
						alpha = self.alpha,is_training = is_training)
						
		if is_training:
			#训练时，实际标签的维度为25
			self.labels = tf.placeholder(
				tf.float32,
				[None,self.cell_size,self.cell_size,5 + self.num_class])
			selfs.loss_layer(self.logits,self.labels)
			self.total_loss = tf.losses.get_total_loss()
			tf.summary.scalar('total_loss',self.total_loss)
			
	def build_network(self,
					images,
					num_outputs,
					alpha,
					keep_prob = 0.5,
					is_training = True,
					scope = 'yolo'):
		
		







