"""
CUDA_VISIBLE_DEVICES=1 python -i training.py \
--data_path=/home/siit/navi/data/input_data/cifar/ \
--meta_path=/home/siit/navi/data/meta_data/cifar/ \
--n_classes=10 --im_size=224 --batch_size=10 \
--label_processed True \

"""
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, dest='data_path', default='/home/siit/navi/data/input_data/mscoco/')
parser.add_argument('--meta_path', type=str, dest='meta_path', default='/home/siit/navi/data/meta_data/mscoco/')
parser.add_argument('--sample_path', type=str, dest='sample_path', default='./sample')
parser.add_argument('--model_path', type=str, dest='model_path', default='/shared/data/models/')
parser.add_argument('--epoch', type=int, dest='epoch', default=1000)

parser.add_argument('--n_classes', type=int, dest='n_classes', default=10)
parser.add_argument('--im_size', type=int, dest='im_size', default=64)
parser.add_argument('--ratio', type=int, dest='ratio', default=2)
parser.add_argument('--batch_size', type=int, dest='batch_size', default=16)

parser.add_argument('--label_processed', type=bool, dest='label_processed', default=True)
parser.add_argument('--save_freq', type=int, dest='save_freq', default=1000)
parser.add_argument('--print_freq', type=int, dest='print_freq', default=50)
parser.add_argument('--memory_usage', type=float, dest='memory_usage', default=0.96)

parser.add_argument('--mode', type=str, dest='mode', default='pretrained')
parser.add_argument('--load_checkpoint', type=bool, dest='load_checkpoint', default=False)
parser.add_argument('--checkpoint_path', type=str, dest='checkpoint_path', default='./checkpoints')
config, unparsed = parser.parse_known_args() 

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.memory_usage)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

import os, random
import data_loader
import numpy as np
import pdb
from model import *

# -------------------- Model -------------------- #
depth = 3
window = 3
height = config.im_size
width = config.im_size
channels = 3
im_size = [height, width, channels]
target_size = [config.ratio * height, config.ratio * width, channels]

model = SISR(sess, config, 'SISR')

# -------------------- Data maniging -------------------- #

train_LR_files = [os.path.join(dp, f)
		for dp, dn, filenames in os.walk(config.data_path)
		for f in filenames if 'train/LR' in dp]

# list_files.sort()

batch_size = config.batch_size


# -------------------- Training -------------------- #

# If you want to debug the model, write the following command on the console
# log_ = model.sess.run([model.logits], feed_dict={model.X: Xbatch, model.Y: Ybatch, model.training: True})

counter = 0
for epoch in range(config.epoch):

	random.shuffle(train_LR_files)

	num_file = len(train_LR_files)

	if num_file ==0:
		break

	total_batch = int(num_file / batch_size)
	total_cost = 0
	final_acc = 0

	for i in range(total_batch):
		# Get the batch as [batch_size, 28,28] and [batch_size, n_classes] ndarray

		Xbatch = data_loader.queue_data_dict(
			train_LR_files[i*batch_size:(i+1)*batch_size], im_size, config.label_processed)

		train_HR_files = [os.path.join(config.data_path, 'train/HR', os.path.basename(file_path))
			for file_path in train_LR_files[i*batch_size:(i+1)*batch_size]]
		
		Ybatch = data_loader.queue_data_dict(
			train_HR_files, target_size, config.label_processed)

		_, cost_val = model.train(Xbatch, Ybatch)

		total_cost += cost_val

		counter += 1

		if np.mod(counter, config.print_freq) == 0:
			print('Step:', '%05dk' % (counter),
				'\tAvg. cost =', '{:.5f}'.format(cost_val))

		# Save the model
		if np.mod(counter, config.save_freq) == 0:
			if not os.path.exists(config.checkpoint_path):
				os.mkdir(config.checkpoint_path)
			model.saver.save(sess, os.path.join(config.checkpoint_path, 
				'vgg19_{0:03d}k'.format(int(counter/1000))))
			print('Model ')
	

def visualize(self, input_files, target_files, sample_dir, counter, is_testing = False, args = None):

	num_input = 4
	num_col = 3
	fig=plt.figure(figsize=(8, 8))

	for i in range(num_input):

		input_files = list(dataA[(self.batch_size)*i:(self.batch_size)*(i+1)])
		sample_images = [load_test_data(input_file, arg.im_size) for input_file in input_files]
		sample_images = np.array(sample_images).astype(np.float32)
		#pdb.set_trace()

		# fake_A, fake_B, rec_A, rec_B = self.sess.run([self.fake_A, self.fake_B, self.fake_A_, self.fake_B_], feed_dict={self.real_data: sample_images})
		OtoT = self.sess.run(A2B, feed_dict={input_A: sample_images})
		OtoTtoO = self.sess.run(B2A, feed_dict={input_B: OtoT})
		fig.add_subplot(num_input, num_col, num_col*i+1)
		plt.imshow((sample_images[0,:,:,:3]+1)/2)
		fig.add_subplot(num_input, num_col, num_col*i+2)
		plt.imshow((OtoT[0,:,:,:3]+1)/2)
		key_layer = np.repeat(np.expand_dims(OtoT[0,:,:,-1], axis=-1), 3, axis=2)

		fig.add_subplot(num_input, num_col, num_col*i+3)
		plt.imshow((OtoT[0,:,:,:3]+key_layer+2)/4)
		fig.add_subplot(num_input, num_col, num_col*i+4)
		plt.imshow((OtoTtoO[0,:,:,:3]+1)/2)


	plt.savefig(os.path.join(sample_dir, 'A_{0:06d}.jpg'.format(int(counter/self.print_freq))))
# -------------------- Testing -------------------- #

