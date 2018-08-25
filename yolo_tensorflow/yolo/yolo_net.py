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
	
	def __init__(self, is_training=True):
		self.classes = cfg.CLASSES #目标类别
		self.num_class = len(self.classes) #类别数目
		self.image_size = cfg.IMAGE_SIZE #输入图像的大小
		self.cell_size = cfg.CELL_SIZE #cell的大小
		self.boxes_per_cell = cfg.BOXES_PER_CELL #每个cell负责的box数目
		self.output_size = (self.cell_size * self.cell_size) * \
			(self.num_class + self.boxes_per_cell * 5) #输出数据的维度：7*7*(20 + 2 * 5) = 1470
			
		self.scale = 1.0 * self.image_size / self.cell_size #缩放比
		
		#7*7个cell属于20个物体类别的概率 + 98个box  边界
		self.boundary1 = self.cell_size * self.cell_size * self.num_class 
		self.boundary2 = self.boundary1 +\
			self.cell_size * self.cell_size * self.boxes_per_cell
			
		
		self.object_scale = cfg.OBJECT_SCALE #值为1，存在目标的因子
		self.noobject_scale = cfg.NOOBJECT_SCALE #值为1，不存在目标的因子
		self.class_scale = cfg.CLASS_SCALE #类别损失函数的因子
		self.coord_scale = cfg.COORD_SCALE #坐标损失函数的因子
		
		self.learning_rate = cfg.LEARNING_RATE #学习速率
		self.batch_size = cfg.BATCH_SIZE #批次大小
		self.alpha = cfg.ALPHA #alpha
		
		#[2,7,7] -> [7,7,2]
		self.offset = np.transpose(np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
            (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))
		
		tf.reset_default_graph()	
		#输入变量 448x448x3 
		self.images = tf.placeholder(
            tf.float32, [None, self.image_size, self.image_size, 3],
            name='images')
		
		#构建网络图,返回预测结果
		self.logits = self.build_network(self.images, num_outputs=self.output_size, alpha=self.alpha,is_training=is_training)
						
		if is_training:
			#训练时，实际标签的维度为25
		    self.labels = tf.placeholder(
				tf.float32,
				[None, self.cell_size, self.cell_size, 5 + self.num_class])
		    self.loss_layer(self.logits, self.labels)
		    self.total_loss = tf.losses.get_total_loss()
		    tf.summary.scalar('total_loss', self.total_loss)
	
	#构造网络图
	def build_network(self,
                      images,
                      num_outputs,
                      alpha,
                      keep_prob=0.5,
                      is_training=True,
                      scope='yolo'):
		
		#tf.reset_default_graph()
		with tf.variable_scope(scope):
			with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                activation_fn=leaky_relu(alpha),
                weights_regularizer=slim.l2_regularizer(0.0005),
                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)
            ):
				net = tf.pad(\
                images, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]),
                name='pad_1') #对输入数据的宽高进行填充，batch_size和channel不做填充
				net = slim.conv2d(\
                net, 64, 7, 2, padding='VALID', scope='conv_2') #conv：64个7x7的卷积核，以2为步伐进行滤波,out:224x224*64
				net = slim.max_pool2d(net,2,padding = 'SAME',scope = 'pool_3') #pool:最大池化kernel=2,stride = 2,out:112x112x64
				net = slim.conv2d(net,192,3,scope = 'conv_4') #conv: num_kernel = 192,kernel_size=3,out:112x112x192 
				net = slim.max_pool2d(net,2,padding = 'SAME',scope = 'pool_5') #pool: kernel_size = 2,stride = 2,out:56x56x192
				net = slim.conv2d(net,128,1,scope = 'conv_6') #conv: num_kernel = 128,kernel_size = 1,out:56x56x128
				net = slim.conv2d(net,256,3,scope = 'conv_7') #conv: num_kernel = 256,kernel_size = 3,out:56x56x256
				net = slim.conv2d(net,256,1,scope = 'conv_8') #conv: num_kernel = 256,kernel_size = 1,out:56x56x256
				net = slim.conv2d(net,512,3,scope = 'conv_9') #conv: num_kernel = 512,kernel_size = 3,out:56x56x512
				net = slim.max_pool2d(net,2,padding = 'SAME',scope = 'pool_10') #pool: kernel_size = 2,stride = 2,out:28x28x512
				net = slim.conv2d(net,256,1,scope = 'conv_11') #conv: num_kernel = 256,kernel_size = 1,out:28x28x256
				net = slim.conv2d(net,512,3,scope = 'conv_12') #conv: num_kernel = 512,kernel_size = 3,out:28x28x512
				net = slim.conv2d(net,256,1,scope = 'conv_13') #conv: num_kernel = 256,kernel_size = 1,out:28x28x256
				net = slim.conv2d(net,512,3,scope = 'conv_14') #conv: num_kernel = 512,kernel_size = 3,out:28x28x512
				net = slim.conv2d(net,256,1,scope = 'conv_15') #conv: num_kernel = 256,kernel_size = 1,out:28x28x256
				net = slim.conv2d(net,512,3,scope = 'conv_16') #conv: num_kernel = 512,kernel_size = 3,out:28x28x512
				net = slim.conv2d(net,256,1,scope = 'conv_17') #conv: num_kernel = 256,kernel_size = 1,out:28x28x256
				net = slim.conv2d(net,512,3,scope = 'conv_18') #conv: num_kernel = 512,kernel_size = 3,out:28x28x512
				net = slim.conv2d(net,512,1,scope = 'conv_19') #conv: num_kernel = 512,kernel_size = 1,out:28x28x512
				net = slim.conv2d(net,1024,3,scope = 'conv_20') #conv: num_kernel = 1024,kernel_size = 3,out:28x28x1024
				net = slim.max_pool2d(net,2,padding = 'SAME',scope = 'pool_21') #pool:kernel_size = 2,stride = 2,out: 14x14x1024
				net = slim.conv2d(net,512,1,scope = 'conv_22') #conv: num_kernel = 512,kernel_size = 1,out: 14x14x512
				net = slim.conv2d(net,1024,3,scope = 'conv_23') #conv: num_kernel = 1024,kernel_size = 3,out:14x14x1024
				net = slim.conv2d(net,512,1,scope = 'conv_24') #conv: num_kernel = 512,kernel_size = 1,out: 14x14x512
				net = slim.conv2d(net,1024,3,scope = 'conv_25') #conv: num_kernel = 1024,kernel_size = 3,out:14x14x1024
				net = slim.conv2d(net,1024,3,scope = 'conv_26') #conv: num_kernel = 1024,kernel_size = 3,out:14x14x1024
				net = tf.pad(net,np.array([[0,0],[1,1],[1,1],[0,0]]),name = 'pad_27') #对特征图进行填充
				net = slim.conv2d(net,1024,3,2,padding = 'VALID',scope = 'conv_28') #conv: num_kernel = 1024, kernel_size = 3,stride = 2,out:7x7x1024
				net = slim.conv2d(net,1024,3,scope = 'conv_29') #conv: num_kernel = 1024,kernel_size = 3,out:7x7x1024
				net = slim.conv2d(net,1024,3,scope = 'conv_30') #conv: num_kernel = 1024,kernel_size = 3,out:7x7x1024
				net = tf.transpose(net,[0,3,1,2],name = 'trans_31') #[batchsize channel 7 7]
				net = slim.flatten(net,scope = 'flat_32') # 
				net = slim.fully_connected(net,512,scope = 'fc_33') #1x512
				net = slim.fully_connected(net,4096,scope = 'fc_34') #1x4096
				net = slim.dropout(
					net,keep_prob = keep_prob,is_training = is_training,
					scope = 'dropout_35') #dropout
				net = slim.fully_connected(net,num_outputs,activation_fn = None,scope = 'fc_36') #1470
					
			return net
		
	def calc_iou(self,boxes1,boxes2,scope = 'iou'):
		
		with tf.variable_scope(scope):
			#transform (x_center,y_center,w,h) to (x1,y1,x2,y2)
			boxes1_t = tf.stack([boxes1[...,0] - boxes1[...,2] / 2.0, #x - w / 2
								boxes1[...,1] - boxes1[...,3] / 2.0,  #y - h / 2
								boxes1[...,0] + boxes1[...,2] / 2.0,  #x + w / 2
								boxes1[...,1] + boxes1[...,3] / 2.0], #y + h / 2
								axis = -1)
								
			boxes2_t = tf.stack([boxes2[...,0] - boxes2[...,2] / 2.0,
								boxes2[...,1] - boxes2[...,3] / 2,
								boxes2[...,0] + boxes2[...,2] / 2,
								boxes2[...,1] + boxes2[...,3] / 2],
								axis = -1)
	
			#calculate the left up point and right down point
			lu = tf.maximum(boxes1_t[...,:2],boxes2_t[...,:2])
			rd = tf.minimum(boxes1_t[...,2:],boxes2_t[...,2:])
			
			#intersection
			intersection = tf.maximum(0,rd - lu)
			inter_square = intersection[...,0] * intersection[...,1]
			
			#calculate the boxs1 square and box2 square
			square1 = boxes1[...,2] * boxes1[...,3]
			square2 = boxes2[...,2] * boxes2[...,3]
			
			#计算并集
			union_square = tf.maximum(square1 + square2 - inter_square,1e-10)
		return tf.clip_by_value(inter_square / union_square,0.0,1.0)
		
		
	def loss_layer(self,predicts,labels,scope = 'loss_layer'):
		with tf.variable_scope(scope):
			predict_classes = tf.reshape(predicts[:,:self.boundary1],[self.batch_size,self.cell_size,self.cell_size,self.num_class])#网络输出端类别数据
			predict_scales = tf.reshape(predicts[:,self.boundary1:self.boundary2],[self.batch_size,self.cell_size,self.cell_size,self.boxes_per_cell])#网络输出端置信度
			predict_boxes = tf.reshape(predicts[:,self.boundary2:],[self.batch_size,self.cell_size,self.cell_size,self.boxes_per_cell,4])#网络输出端box数据
			
			#label存储信息的格式：[response boxes classes]
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
			
			#calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
			object_mask = tf.reduce_max(iou_predict_truth,3,keep_dims = True) #取第三个维度上的最大值，并保持第三个维度的大小
			object_mask = tf.cast((iou_predict_truth >= object_mask),tf.float32) * response #表示包含目标物体的位置为1，其他位置为0
			
			#calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
			noobject_mask = tf.ones_like(object_mask,dtype = tf.float32) - object_mask #计算无目标物体的掩膜
			
			boxes_tran = tf.stack([
			boxes[...,0] * self.cell_size - offset,
			boxes[...,1] * self.cell_size - offset_tran,
			tf.sqrt(boxes[...,2]),
			tf.sqrt(boxes[...,3])],axis = -1)
			
			#class_loss 类别损失函数
			class_delta = response * (predict_classes - classes)  #有目标情况下，类别误差
			class_loss = tf.reduce_mean(
				tf.reduce_sum(tf.square(class_delta),axis = [1,2,3]),name = 'class_loss') * self.class_scale
				
			#object_loss 目标物体confidence的损失
			object_delta = object_mask * (predict_scales - iou_predict_truth)
			object_loss = tf.reduce_mean(
				tf.reduce_sum(tf.square(object_delta),axis = [1,2,3]),name = 'object_loss') * self.object_scale
			
			#noobject_delta
			noobject_delta = noobject_mask * predict_scales
			noobject_loss = tf.reduce_mean(
				tf.reduce_sum(tf.square(noobject_delta),axis = [1,2,3]),
				name = 'noobject_loss') * self.noobject_scale
				
			#coord_loss
			coord_mask = tf.expand_dims(object_mask,4) #先扩维
			boxes_delta = coord_mask * (predict_boxes - boxes_tran)
			coord_loss = tf.reduce_mean(
				tf.reduce_sum(tf.square(boxes_delta),axis = [1,2,3,4]),
				name = 'coord_loss') * self.coord_scale
			
			#total loss
			tf.losses.add_loss(class_loss)
			tf.losses.add_loss(object_loss)
			tf.losses.add_loss(noobject_loss)
			tf.losses.add_loss(coord_loss)
			
			tf.summary.histogram('boxes_delta_x',boxes_delta[...,0])
			tf.summary.histogram('boxes_delta_y',boxes_delta[...,1])
			tf.summary.histogram('boxes_delta_w',boxes_delta[...,2])
			tf.summary.histogram('boxes_delta_h',boxes_delta[...,3])
			tf.summary.histogram('iou',iou_predict_truth)

#leaky 激活函数			
def leaky_relu(alpha):
	def op(inputs):
		return tf.nn.leaky_relu(inputs,alpha = alpha,name = 'leaky_relu')
	return op







