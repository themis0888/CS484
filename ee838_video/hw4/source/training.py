"""
CUDA_VISIBLE_DEVICES=0 python -i training.py \
--data_path=/home/siit/navi/data/input_data/60fps/ \
--im_size=128 --batch_size=4 --ratio=2 \
--mode=training --checkpoint_path=./01checkpoints \
--model_mode=cbsr --sample_path=1127cbsr_tr \

CUDA_VISIBLE_DEVICES=0 python -i training.py \
--data_path=/home/siit/navi/data/input_data/60fps/ \
--im_size=128 --batch_size=4 --ratio=2 \
--mode=testing --checkpoint_path=./01checkpoints \
--model_mode=cbsr --sample_path=1127cbsr_te \

"""
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, dest='data_path', default='/navi/data/input_data/mscoco/')
parser.add_argument('--meta_path', type=str, dest='meta_path', default='/navi/data/meta_data/mscoco/')
parser.add_argument('--sample_path', type=str, dest='sample_path', default='./sample')
parser.add_argument('--log_path', type=str, dest='log_path', default='./log')
parser.add_argument('--model_path', type=str, dest='model_path', default='/shared/data/models/')
parser.add_argument('--epoch', type=int, dest='epoch', default=1000)

parser.add_argument('--n_classes', type=int, dest='n_classes', default=10)
parser.add_argument('--resize', type=lambda x: x.lower() in ('true', '1'), dest='resize', default=False)
parser.add_argument('--im_size', type=int, dest='im_size', default=64)
parser.add_argument('--ratio', type=int, dest='ratio', default=2)
parser.add_argument('--lr', type=float, dest='lr', default=1e-4)
parser.add_argument('--batch_size', type=int, dest='batch_size', default=16)
parser.add_argument('--data_mean', type=list, default=[0.28405, 0.32267, 0.28169])
parser.add_argument('--model_mode', type=str, dest='model_mode', default='cbsr')

parser.add_argument('--label_processed', type=bool, dest='label_processed', default=True)
parser.add_argument('--save_freq', type=int, dest='save_freq', default=1000)
parser.add_argument('--print_freq', type=int, dest='print_freq', default=50)
parser.add_argument('--memory_usage', type=float, dest='memory_usage', default=0.45)

parser.add_argument('--re_train', type=lambda x: x.lower() in ('true', '1'), dest='re_train', default=False)
parser.add_argument('--mode', type=str, dest='mode', default='training')
parser.add_argument('--load_checkpoint', type=bool, dest='load_checkpoint', default=False)
parser.add_argument('--checkpoint_path', type=str, dest='checkpoint_path', default='./checkpoints')
config, unparsed = parser.parse_known_args() 

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.memory_usage)
sess = tf.InteractiveSession()#config=tf.ConfigProto(gpu_options=gpu_options))

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file as ptck
import os, random
import numpy as np
import cv2
from model import *
import skimage.measure as skms
import data_loader
import pdb

# -------------------- Model -------------------- #
depth = 3
window = 3
height = config.im_size
width = config.im_size
channels = 3
im_size = [height, width, channels]
target_size = [config.ratio * height, config.ratio * width, channels]

# -------------------- Data maniging -------------------- #

if not os.path.exists(config.sample_path):
	os.mkdir(config.sample_path)
if not os.path.exists(config.log_path):
	os.mkdir(config.log_path)



# If you want to debug the model, write the following command on the console
# log_ = model.sess.run([model.logits], feed_dict={model.X: Xbatch, model.Y: Ybatch, model.training: True})

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name



def testing(config, model, train_LR_files):
	data_path = config.data_path
	imsize = config.im_size
	cap = []
	for index in range(1,5):
		cap.append(cv2.VideoCapture(
			os.path.join(data_path, 'video{}.mp4'.format(index))))

	batch_size = 4
	counter = 0
	log_file = open(os.path.join(config.log_path, 'training_log.txt'), 'w')
	for epoch in range(config.epoch):

		num_file = len(train_LR_files)
		if num_file ==0:
			break

		total_batch = int(num_file / batch_size)
		total_cost = 0
		final_acc = 0

		for j in range(total_batch):
			# Get the data batch in [batch_size, im_size] ndarray format.

			index_list = list(range(4))
			random.shuffle(index_list)

			frame1, frame2, frame3 = [], [], []
			# pdb.set_trace()

			for i in range(4):
				ret, temp = cap[i].read()
				frame1.append(np.expand_dims(temp, axis = 0)/255)
				ret, temp = cap[i].read()
				frame2.append(np.expand_dims(temp, axis = 0)/255)
				ret, temp = cap[i].read()
				frame3.append(np.expand_dims(temp, axis = 0)/255)

			Xbatch = np.concatenate((frame1[index_list[0]], frame3[index_list[0]]), axis = 3)
			Ybatch = frame2[index_list[0]]
									

			frame1, frame2, frame3 = [], [], []
			coord = (int(np.random.uniform(0,1)*(720-imsize)), int(np.random.uniform(0,1)*(1280-imsize))) 
			# train
			
			# total_cost += cost_val

			counter += 1

			# visualize the current recovery process and print the log
			if np.mod(counter, 1) == 0:
				
				out_batch = model.visualize(Xbatch, Ybatch, config.sample_path, counter)
				psnr, ssim = 0, 0
				Ybatch = Ybatch.astype('float32')
				
				i = 0
				temp_psnr = skms.compare_psnr(Ybatch[i], out_batch[i])
				temp_ssim = skms.compare_ssim(Ybatch[i], out_batch[i], multichannel=True)
				psnr += temp_psnr
				ssim += temp_ssim

				log = ('Step: {:05d}'.format(counter) +
					'\tPSNR: {:2.1f}'.format(psnr) +
					'\tSSIM: {:.3f}'.format(ssim))
				print(log)
				log_file.write(log + '\n')

	


def training(config, model, train_LR_files):
	data_path = config.data_path
	imsize = config.im_size
	cap = []
	for index in range(1,5):
		cap.append(cv2.VideoCapture(
			os.path.join(data_path, 'video{}.mp4'.format(index))))

	batch_size = 4
	counter = 0
	log_file = open(os.path.join(config.log_path, 'training_log.txt'), 'w')
	for epoch in range(config.epoch):

		num_file = len(train_LR_files)
		if num_file ==0:
			break

		total_batch = int(num_file / batch_size)
		total_cost = 0
		final_acc = 0

		for j in range(total_batch):
			# Get the data batch in [batch_size, im_size] ndarray format.

			index_list = list(range(4))
			random.shuffle(index_list)

			frame1, frame2, frame3 = [], [], []
			# pdb.set_trace()

			for i in range(4):
				ret, temp = cap[i].read()
				frame1.append(np.expand_dims(temp, axis = 0)/255)
				ret, temp = cap[i].read()
				frame2.append(np.expand_dims(temp, axis = 0)/255)
				ret, temp = cap[i].read()
				frame3.append(np.expand_dims(temp, axis = 0)/255)

			Xbatch = np.concatenate((np.concatenate((frame1[index_list[0]], frame3[index_list[0]]), axis = 3),
									np.concatenate((frame1[index_list[1]], frame3[index_list[1]]), axis = 3),
									np.concatenate((frame1[index_list[2]], frame3[index_list[2]]), axis = 3),
									np.concatenate((frame1[index_list[3]], frame3[index_list[3]]), axis = 3)), axis = 0)
			Ybatch = np.concatenate((frame2[index_list[0]],
									frame2[index_list[1]], 
									frame2[index_list[2]], 
									frame2[index_list[3]]), axis = 0)

			frame1, frame2, frame3 = [], [], []
			coord = (int(np.random.uniform(0,1)*(720-imsize)), int(np.random.uniform(0,1)*(1280-imsize))) 
			Xbatch = Xbatch[:, coord[0]:coord[0]+128, coord[1]:coord[1]+128, :]
			Ybatch = Ybatch[:, coord[0]:coord[0]+128, coord[1]:coord[1]+128, :]
			# train
			_, cost_val = model.train(Xbatch, Ybatch)
			total_cost += cost_val

			counter += 1

			# visualize the current recovery process and print the log
			if np.mod(counter, config.print_freq) == 0:
				
				out_batch = model.visualize(Xbatch, Ybatch, config.sample_path, counter)
				psnr, ssim = 0, 0
				Ybatch = Ybatch.astype('float32')
				
				for i in range(config.batch_size):
					# pdb.set_trace()
					temp_psnr = skms.compare_psnr(Ybatch[i], out_batch[i])
					temp_ssim = skms.compare_ssim(Ybatch[i], out_batch[i], multichannel=True)
					psnr += temp_psnr
					ssim += temp_ssim

				log = ('Step: {:05d}'.format(counter) +
					'\tCost: {:.3f}'.format(cost_val) +
					'\tPSNR: {:2.1f}'.format(psnr/config.batch_size) +
					'\tSSIM: {:.3f}'.format(ssim/config.batch_size))
				print(log)
				log_file.write(log + '\n')

			# Save the model
			if np.mod(counter, config.save_freq) == 0:
				if not os.path.exists(config.checkpoint_path):
					os.mkdir(config.checkpoint_path)
				model.saver.save(sess, os.path.join(config.checkpoint_path, 
					'vgg19_{0:03d}k'.format(int(counter/1000))))
				print('Model ')
	
	

if config.mode == 'training':
	# -------------------- Training -------------------- #
	model = Interpolation(sess, config, 'SISR')
	training(config, model, list(range(50000)))
	

	# -------------------- Testing -------------------- #
elif config.mode == 'testing':
	model = Interpolation(sess, config, 'SISR')
	testing(config, model, list(range(50000)))



