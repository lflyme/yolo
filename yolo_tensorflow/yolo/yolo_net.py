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
			(self.num_class + self.boxes_per_cell * 5) #7*7*(20 + 2 * 5)
			
		self.scale = 1.0 * self.image_size / self.cell_size #缩放比
		self.boundary1 = self.cell_size * self.cell_size * self.num_class #类别的维度边界
		self.boundary2 = self.boundary1 +\
			self.cell_size * self.cell_size * self.boxes_per_cell
			#box维度的边界
		
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
		
		with tf.variable_scope(scope):
			with slim.arg_scope(
				[slim.conv2d,slim.fully_connected],
				actiivation_fn = leaky_relu(alpha),
				weights_regularizer = slim.l2_regularizer(0.0005),
				weights_initializer = tf.truncated_normal_initializer(0.0,0.01)):
					net = tf.pad(
						images,np.array([[0,0],[3,3],[3,3],[0,0]]),
						name = 'pad_1') #对输入数据的宽高进行拓展
					net = slim.conv2d(net,64,7,2,padding = 'VALID',scope = 'conv_2') #卷积：64个7x7的卷积核，以2为步伐进行滤波
					net = slim.max_pool2d(net,2,padding = 'SAME',scope = 'pool_3')
					net = slim.conv2d(net,192,3,scope = 'conv_4')
					net = slim.max_pool2d(net,2,padding = 'SAME',scope = 'pool_5')
					net = slim.conv2d(net,128,1,scope = 'conv_6')
					net = slim.conv2d(net,256,3,scope = 'conv_7')
					net = slim.conv2d(net,256,1,scope = 'conv_8')
					net = slim.conv2d(net,512,3,scope = 'conv_9')
					net = slim.max_pool2d(net,2,padding = 'SAME',scope = 'pool_10')
					net = slim.conv2d(net,256,1,scope = 'conv_11')
					net = slim.conv2d(net,512,3,scope = 'conv_12')
					net = slim.conv2d(net,256,1,scope = 'conv_13')
					net = slim.conv2d(net,512,3,scope = 'conv_14')
					net = slim.conv2d(net,256,1,scope = 'conv_15')
					net = slim.conv2d(net,512,3,scope = 'conv_16')
					net = slim.conv2d(net,256,1,scope = 'conv_17')
					net = slim.conv2d(net,512,3,scope = 'conv_18')
					net = slim.conv2d(net,512,1,scope = 'conv_19')
					net = slim.conv2d(net,1024,3,scope = 'conv_20')
					net = slim.max_pool2d(net,2,padding = 'SAME',scope = 'pool_21')
					net = slim.conv2d(net,512,1,scope = 'conv_22')
					net = slim.conv2d(net,1024,3,scope = 'conv_23')
					net = slim.conv2d(net,512,1,scope = 'conv_24')
					net = slim.conv2d(net,1024,3,scope = 'conv_25')
					net = slim.conv2d(net,1025,3,scope = 'conv_26')
					net = tf.pad(net,np.array([[0,0],[1,1],[1,1],[0,0]]),name = 'pad_27')
					net = slim.conv2d(net,1024,3,2,padding = 'VALID',scope = 'conv_28')
					net = slim.conv2d(net,1024,3,scope = 'conv_29')
					net = slim.conv2d(net,1024,3,scope = 'conv_30')
					net = tf.transpose(net,[0,3,1,2],name = 'trans_31')
					net = slim.flatten(net,scope = 'flat_32')
					net = slim.fully_connected(net,512,scope = 'fc_33')
					net = slim.fully_connected(net,4096,scope = 'fc_34')
					net = slim.dropout(
						net,keep_prob = keep_prob,is_training = is_training,
						scope = 'dropout_35')
					net = slim.fully_connected(net,num_outputs,activation_fn = None,scope = 'fc_36')
					
			return net
		
	def calc_iou(self,boxes1,boxes2,scope = 'iou'):
		
	



	def loss_layer(self,predicts,labels,scope = 'loss_layer'):
		with tf.variable_scope(scope):
			predict_classes = tf.reshape(predicts[:,:self.boundary1],[self.batch_size,self.cell_size,self.cell_size,self.num_class])#网络输出端类别数据
			predict_scales = tf.reshape(predicts[:,self.boundary1:self.boundary2],[self.batch_size,self.cell_size,self.cell_size,self.boxes_per_cell])#网络输出端置信度
			predict_boxes = tf.reshape(predicts[:,self.boundary2:],[self.batch_size,self.cell_size,self.cell_size,self.boxes_per_cell,4])#网络输出端box数据
			
			response = tf.reshape(labels[...,0],[self.batch_size,self.cell_size,self.cell_size,1])#标签中置信度
			boxes = tf.reshape(labels[...,1:5],[self.batch_size,self.cell_size,self.cell_size,1,4])#标签中box
			boxes = tf.tile(boxes,[1,1,1,self.boxes_per_cell,1] / self.image_size)#将label中的box格式转换为与predict中box对应的格式
			classes = labels[...,5:]#标签中的类别信息
			
			offset = tf.reshape(tf.constant(self.offset,dtype = tf.float32),[1,self.cell_size,self.cell_size,self.boxes_per_cell])
			offset = tf.tile(offset,[self.batch_size,1,1,1])#将offset复制batch_size份
			offset_tran = tf.transpose(offset,(0,2,1,3))
			predict_boxes_tran = tf.stack([(predict_boxes[...,0] + offset) / self.cell_size,
									(predict_boxes[...,1] + offset_tran) / self.cell_size,
									tf.square(predict_boxes[...,2]),
									tf.square(predict_boxes[...,3])],axis = -1)
									
			iou_predict_truth = self.calc_iou(predict_boxes_tran,boxes)
			
			
			
					
		
		







