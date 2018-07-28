import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
import yolo.config as cfg
from yolo.yolo_net import YOLONet
from utils.timer import Timer

class Detector(object):

	def __init__(self,net,weights_file):
		self.net = net#加载网络图
		self.weights_file = weights_file#加载权重
		self.classes = cfg.CLASSES#加载类别
		self.num_class = len(self.classes)#一共有多少类
		self.image_size = cfg.IMAGE_SIZE#图片的大小
		self.cell_size = cfg.CELL_SIZE #没个cell的大小
		self.boxes_per_cell = cfg.BOXES_PER_CELL #没个cell有多少个boxes
		self.threshold = cfg.THRESHOLD
		self.iou_threshold = cfg.IOU_THRESHOLD
		self.boundary1 = self.cell_size * self.cell_size * self.num_class #类别向量维度的边界
		self.boundary2 = self.boundary1 + \
			self.cell_size * self.cell_size * self.boxes_per_cell #box向量维度的边界
		self.sess= tf.Session() #创建会话
		self.sess.run(tf.global_variables_initalizer()) #图中所有变量的初始化
		
		#加载模型文件
		print('Restoring weights from: ' + self.weights_file)
		self.saver = tf.train.Saver()
		self.saver.restore(self.sess,self.weights_file)
	
	def detect(self,img):
		img_h,img_w,_ = img.shape
		inputs= cv2.resize(img,(self.img_size,self.image_size))
		inputs = cv2.cvtColor(inputs,cv2.COLOR_BGR2RGB).astype(np.float32) #bgr -> rgb
		inputs = (inputs / 255.0) * 2.0 - 1.0
		inputs = np.reshape(inputs,(1,self.image_size,self.image_size,3))
		
		result = self.detect_from_cvmat(inputs)[0]
		
		for i in range(len(result)):
			result[i][1] *= (1.0 * img_w / self.image_size)
			result[i][2] *= (1.0 * img_h / self.image_size)
			result[i][3] *= (1.0 * img_w / self.image_size)
			result[i][4] *= (1.0 * img_h / self.image_size)
		
		return result
	
	def detect_from_cvmat(self,inputs):
		net_output = self.sess.run(self.net.logits,
							feed_dict = {self.net.images:inputs})
							
		results = []
		for i in range(net_output.shape[0]):
			results.append(self.interpret_output(net_output[i]))
			
		return results
		
	
		
		





