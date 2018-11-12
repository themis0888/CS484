"""
CUDA_VISIBLE_DEVICES=0 python -i training.py \
--data_path=/home/siit/navi/data/input_data/ee838_hw3/ \
--im_size=256 --batch_size=8 --ratio=1 \
--mode=training --checkpoint_path=./01checkpoints \

CUDA_VISIBLE_DEVICES=0 python -i training.py \
--data_path=/home/siit/navi/data/input_data/ee838_hw3/ \
--batch_size=16 --ratio=1 --sample_path=01sample \
--mode=testing --checkpoint_path=./02checkpoints \
--resize true

"""
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, dest='data_path', default='/home/siit/navi/data/input_data/ee838_hw3/')
parser.add_argument('--meta_path', type=str, dest='meta_path', default='/home/siit/navi/data/meta_data/ee838_hw3/')
parser.add_argument('--sample_path', type=str, dest='sample_path', default='./sample')
parser.add_argument('--log_path', type=str, dest='log_path', default='./log')
parser.add_argument('--model_path', type=str, dest='model_path', default='/shared/data/models/')
parser.add_argument('--epoch', type=int, dest='epoch', default=1000)

parser.add_argument('--n_classes', type=int, dest='n_classes', default=10)
parser.add_argument('--resize', type=lambda x: x.lower() in ('true', '1'), dest='resize', default=False)
parser.add_argument('--im_size', type=int, dest='im_size', default=64)
parser.add_argument('--ratio', type=int, dest='ratio', default=2)
parser.add_argument('--lr', type=float, dest='lr', default=0.001)
parser.add_argument('--batch_size', type=int, dest='batch_size', default=16)

parser.add_argument('--label_processed', type=bool, dest='label_processed', default=True)
parser.add_argument('--save_freq', type=int, dest='save_freq', default=200)
parser.add_argument('--print_freq', type=int, dest='print_freq', default=50)
parser.add_argument('--memory_usage', type=float, dest='memory_usage', default=0.45)

parser.add_argument('--re_train', type=lambda x: x.lower() in ('true', '1'), dest='re_train', default=False)
parser.add_argument('--mode', type=str, dest='mode', default='training')
parser.add_argument('--load_checkpoint', type=bool, dest='load_checkpoint', default=False)
parser.add_argument('--checkpoint_path', type=str, dest='checkpoint_path', default='./checkpoints')
config, unparsed = parser.parse_known_args() 

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.memory_usage)
sess = tf.InteractiveSession()#config=tf.ConfigProto(gpu_options=gpu_options))

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file as ptck
import os, random, cv2
import numpy as np
from model import *
import skimage.measure as skms
import scipy.misc
import data_loader
import skimage.io as skio
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

train_LR_files = [os.path.join(dp, f)
		for dp, dn, filenames in os.walk(config.data_path)
		for f in filenames if ('train' in dp) and ('blur' in dp)]

test_LR_files = [os.path.join(dp, f)
		for dp, dn, filenames in os.walk(config.data_path)
		for f in filenames if ('test' in dp) and ('blur' in dp)]
test_LR_files.sort()

batch_size = config.batch_size

# If you want to debug the model, write the following command on the console
# log_ = model.sess.run([model.logits], feed_dict={model.X: Xbatch, model.Y: Ybatch, model.training: True})

if config.mode == 'training':
	# -------------------- Training -------------------- #
	model = Deblur(sess, config, 'Deblur')
	print('Model built')
	counter = 0
	log_file = open(os.path.join(config.log_path, 'training_log.txt'), 'w')
	for epoch in range(config.epoch):

		num_file = len(train_LR_files)
		if num_file ==0:
			break

		total_batch = int(num_file / batch_size)
		total_cost = 0
		final_acc = 0

		for i in range(total_batch):
			# Get the data batch in [batch_size, im_size] ndarray format.
			cord = np.random.rand(2, batch_size)
			Xbatch = data_loader.queue_data_dict(
				train_LR_files[i*batch_size:(i+1)*batch_size], im_size, image_resize = config.resize, crop_cord = cord)

			train_HR_files = [file_path.replace('blur_gamma', 'sharp').replace('blur', 'sharp')
				for file_path in train_LR_files[i*batch_size:(i+1)*batch_size]]
			
			Ybatch = data_loader.queue_data_dict(
				train_HR_files, target_size, image_resize = config.resize, crop_cord = cord)

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
					temp_psnr = skms.compare_psnr(Ybatch[i], out_batch[i])
					temp_ssim = skms.compare_ssim(Ybatch[i], out_batch[i], multichannel=True)
					psnr += temp_psnr
					ssim += temp_ssim

				log = ('Step: {:05d}'.format(counter) +
					'\tCost: {:.3f}'.format(cost_val) +
					'\tmPSNR: {:2.1f}'.format(psnr/config.batch_size))
				print(log)
				log_file.write(log + '\n')

			# Save the model
			if np.mod(counter, config.save_freq) == 0:
				if not os.path.exists(config.checkpoint_path):
					os.mkdir(config.checkpoint_path)
				model.saver.save(sess, os.path.join(config.checkpoint_path, 
					'vgg19_{0:03d}k'.format(int(counter/1000))))
				print('Model ')
	

	# -------------------- Testing -------------------- #
elif config.mode == 'testing':
	model = HDR(sess, config, 'HDR')
	print('The model built')
	log_file = open(os.path.join(config.log_path, 'testing_log.txt'), 'w')
	counter = 0

	num_file = len(test_LR_files)
	batch_size = 1
	total_batch = int(num_file / batch_size)
	total_cost = 0
	final_acc = 0
	total_psnr = 0
	total_ssim = 0
	
	for i in range(total_batch):
		# Get the data batch in [batch_size, im_size] ndarray format.

		Xbatch = data_loader.queue_data_dict(
			test_LR_files[i*batch_size:(i+1)*batch_size], [1088,1920], image_resize= config.resize)
		print(test_LR_files[i*batch_size:(i+1)*batch_size])
		print(Xbatch.shape)
		_, config.height, config.width, _ = Xbatch.shape

		test_HR_files = [os.path.join(config.data_path, 'train/', os.path.basename(file_path))
			for file_path in test_LR_files[i*batch_size:(i+1)*batch_size]]
		
		Ybatch = data_loader.queue_data_dict(
			test_HR_files, [1088,1920],
			image_resize= config.resize)
		
		counter += 1

		out_batch = model.visualize(Xbatch, Ybatch, config.sample_path, counter, save_output = True)
		psnr, ssim = 0, 0
		Ybatch = Ybatch.astype('float32')

		print(test_LR_files[i*batch_size:(i+1)*batch_size][0].split('.')[0])
		for j in range(batch_size):
			"""
			temp_psnr = skms.compare_psnr(Ybatch[j], out_batch[j])
			temp_ssim = skms.compare_ssim(Ybatch[j], out_batch[j], multichannel=True)
			psnr += temp_psnr
			ssim += temp_ssim
			"""
			# pdb.set_trace()
			print(out_batch.shape)
			out_img = scipy.misc.imresize(out_batch[0], (1088, 1920, 3))
			print(out_img.shape)
			skio.imsave(os.path.join(config.sample_path, 
				os.path.basename(test_LR_files[i*batch_size:(i+1)*batch_size][0]).split('.')[0]+'.hdr'), out_img)
			skio.imsave(os.path.join(config.sample_path, 
				os.path.basename(test_LR_files[i*batch_size:(i+1)*batch_size][0])), out_img)
			skio.imsave(os.path.join(config.sample_path, 'HDR_' +
				os.path.basename(test_LR_files[i*batch_size:(i+1)*batch_size][0]).split('.')[0]+'.hdr'), Ybatch[0])
			
		"""
		log = ('File: {:s}'.format(os.path.basename(test_LR_files[i])) +
			'\tmPSNR: {:2.1f}'.format(psnr))
		print(log)
		total_psnr += psnr
		total_ssim += ssim 
		log_file.write(log + '\n')
		"""

	log = ('\nTotal summary \n'+
		'Avg.mPSNR = {:2.2f} \n'.format(total_psnr/num_file))
	print(log)
	log_file.write(log + '\n')
	log_file.close()
	

else:
	model = HDR(sess, config, 'HDR')
	print('The model built')
	log_file = open(os.path.join(config.log_path, config.mode + '_log.txt'), 'w')
	counter = 0
	
	test_LR_files = [os.path.join(dp, f)
		for dp, dn, filenames in os.walk(config.data_path)
		for f in filenames]

	num_file = len(test_LR_files)
	batch_size = 1
	total_batch = int(num_file / batch_size)
	total_cost = 0
	final_acc = 0
	total_psnr = 0
	total_ssim = 0
	
	for i in range(total_batch):
		# Get the data batch in [batch_size, im_size] ndarray format.

		Xbatch = data_loader.queue_data_dict(
			test_LR_files[i*batch_size:(i+1)*batch_size], [1088,1920], image_resize= config.resize)
		print(test_LR_files[i*batch_size:(i+1)*batch_size])
		print(Xbatch.shape)
		
		Ybatch = Xbatch
		
		counter += 1

		out_batch = model.visualize(Xbatch, Ybatch, config.sample_path, counter, save_output = True)
		psnr, ssim = 0, 0
		Ybatch = Ybatch.astype('float32')

		print(test_LR_files[i*batch_size:(i+1)*batch_size][0].split('.')[0])
		for j in range(batch_size):

			# pdb.set_trace()
			print(out_batch.shape)
			# out_img = scipy.misc.imresize(out_batch[0], (1088, 1920, 3))
			out_img = out_batch[0]
			print(out_img.shape)
			skio.imsave(os.path.join(config.sample_path, 'hdr_' +
				os.path.basename(test_LR_files[i*batch_size:(i+1)*batch_size][0])), out_img)
			skio.imsave(os.path.join(config.sample_path, 
				os.path.basename(test_LR_files[i*batch_size:(i+1)*batch_size][0])), Xbatch[0])
			

	log = ('\nTotal summary \n'+
		'Avg.mPSNR = {:2.2f} \n'.format(total_psnr/num_file))
	print(log)
	log_file.write(log + '\n')
	log_file.close()

