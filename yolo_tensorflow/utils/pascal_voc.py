"""
pascal_voc:主要功能
对图像数据进行归一化，同时获取相应的标签数据
"""
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
		self.rebuild = rebuild
		self.cursor = 0
		self.epoch = 1
		self.gt_labels = None
		self.prepare()
	
	#批量读取图片和图片的标签信息
	def get(self):
		images = np.zeros((self.batch_size,self.image_size,self.image_size,3))
		labels = np.zeros((self.batch_size,self.cell_size,self.cell_size,25))
		count = 0
		while count < self.batch_size:
			imname = self.gt_labels[self.cursor]['imname']
			flipped = self.gt_labels[self.cursor]['flipped']
			images[count,:,:,:] = self.image_read(imname,flipped)
			labels[count,:,:,:] = self.gt_labels[self.cursor]['label']
			count += 1
			self.cursor += 1
			if self.cursor >= len(self.gt_labels):
				np.random.shuffle(self.gt_labels)
				self.cursor = 0
				self.epoch += 1
				
		return images,labels
	
	#对图片数据进行格式转换以及归一化等处理
	def image_read(self,imname,flipped = False):
		image = cv2.imread(imname)
		image = cv2.resize(image,(self.image_size,self.image_size))
		image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)
		image = (image / 255.0) * 2.0 - 1.0 #数据归一化到[-1,1]
		
		if flippedd:
			image = image[:,::-1,:]#对数据的x维度进行镜像处理
		return image
	
	#准备好数据的标签信息
	def prepare(self):
		gt_labels = self.load_labels()
		#如果要做镜像处理，则相应的label信息做相应的处理
		if self.flipped:
			print('Appending horizontally-flipped training examples...')
			gt_labels_cp = copy.deepcopy(gt_labels)
			for idx in range(len(gt_labels_cp)):
				gt_labels_cp[idx]['flipped'] = True
				gt_labels_cp[idx]['label'] = gt_labels_cp[idx]['label'][:,::-1,:] #label信息水平方向进行镜像处理
				for i in range(self.cell_size):
					for j in range(self.cell_size):
						if gt_labels_cp[idx]['label'][i,j,0] == 1:
							gt_labels_cp[idx]['label'][i,j,1] = \
								self.image_size - 1 - \
								gt_labels_cp[idx]['label'][i,j,1]
			
			gt_labels += gt_labels_cp
		np.random.shuffle(gt_labels)#打乱数据信息
		self.gt_labels = gt_labels
		return gt_labels
		
		
	#gt_label存储图片路径信息，标签信息，是否镜像信息	
	def load_labels(self):
		cache_file = os.path.join(self.cache_path,'pascal_' + self.phase + '_gt_labels.pkl')
		
		if os.path.isfile(cache_file) and not self.rebuild:
			print('Loading gt_labels from: ' + cache_file)
			with open(cache_file,'rb') as f:
				gt_labels = pickle.load(f)
			return gt_labels
			
		print('processing gt_labels from: ' + self.data_path)
		
		if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
		
		if self.phase == 'train':
			txtname = os.path.join(self.data_path,'ImageSets','Main','trainval.txt')
		else:
			txtname = os.path.join(self.data_path,'ImageSets','Main','test.txt')
			
		with open(txtname,'r') as f:
			self.image_index = [x.strip() for x in f.readlines()]
			
		gt_labels = []
		for index in self.image_index:
			label,num = self.load_pascal_annotation(index)
			if num == 0:
				continue
			imname = os.path.join(self.data_path,'JPEGImages',index + '.jpg')
			gt_labels.append({'imname':imname,
								'label':label,
								'flipped':False})
								
		print('Saving gt_labels to:' + cache_file)
		with open(cache_file,'wb') as f:
			pickle.dump(gt_labels,f)
		return gt_labels
			
		
		
	#获取每一张图像中目标物体的label信息	
	def load_pascal_annotation(self,index):
		
		imname = os.path.join(self.data_path,'JPEGImages',index + '.jpg')
		im = cv2.imread(imname)
		
		#缩放比
		h_ratio = 1.0 * self.image_size / im.shape[0]
		w_ratio = 1.0 * self.image_size / im.shape[1]
		
		label = np.zeros((self.cell_size,self.cell_size,25))
		filename = os.path.join(self.data_path,'Annotations',index + '.xml')
		tree = ET.parse(filename)
		objs = tree.findall('object')#获取目标物体的信息
		
		for obj in objs:
			bbox = obj.find('bndbox')#目标物体的box
			#对目标物体的box进行相应的缩放处理
			x1 = max(min((float(bbox.find('xmin').text) - 1) * w_ratio,self.image_size - 1),0)
			y1 = max(min((float(bbox.find('ymin').text) - 1) * h_ratio,self.image_size - 1),0)
			x2 = max(min((float(bbox.find('xmax').text) - 1) * w_ratio,self.image_size - 1),0)
			y2 = max(min((float(bbox.find('ymax').text) - 1) * h_ratio,self.image_size - 1),0)
			
			cls_ind = self.class_to_ind[obj.find('name').text.lower().strip()]
			boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]#box转换为centerx,centery,w,h
			x_ind = int(boxes[0] * self.cell_size / self.image_size)#将图像坐标转换为以7为单元的坐标
			y_ind = int(boxes[1] * self.cell_size / self.image_size)
			if label[y_ind,x_ind,0] == 1:
				continue
			label[y_ind,x_ind,0] = 1
			label[y_ind,x_ind,1:5] = boxes
			label[y_ind,x_ind,5 + cls_ind] = 1
			
		return label,len(objs)
	
		
		
		
		
		
		
		
		
		


