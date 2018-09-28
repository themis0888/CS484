"""
CUDA_VISIBLE_DEVICES=3 CUDA_CACHE_PATH="/gpu_cache/" \
python feature_extractor.py \
--data_path=/shared/data/sample/ \
--list_path=/shared/data/sample/meta/ \
--model_name=vgg_19


CUDA_VISIBLE_DEVICES=1 python feature_extractor.py \
--data_path=/shared/data/danbooru2017/sample/0000/ \
--list_path=/shared/data/meta/danbooru2017/sample/0000/ \
--model_name=vgg_19
"""

import pdb
import random, time, os, sys
import numpy as np
import scipy
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
import data_loader
import scipy.io as sio
from util import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, dest='data_path', default='/shared/data/danbooru2017/256px/')
parser.add_argument('--data_name', type=str, dest='data_name', default='danbooru')
parser.add_argument('--save_path', type=str, dest='save_path', default='/shared/data/meta/danbooru2017/256px/')
parser.add_argument('--input_list', type=str, dest='input_list', default='path_label_list.txt')
parser.add_argument('--model_path', type=str, dest='model_path', default='/shared/data/models/')
parser.add_argument('--model_name', type=str, dest='model_name', default='vgg_19')

parser.add_argument('--memory_usage', type=float, dest='memory_usage', default=0.96)
parser.add_argument('--n_classes', type=int, dest='n_classes', default=50)
parser.add_argument('--max_iter', type=int, dest='max_iter', default=300000)
parser.add_argument('--batch_size', type=int, dest='batch_size', default=1)
parser.add_argument('--train_display', type=int, dest='train_display', default=200)
parser.add_argument('--val_display', type=int, dest='val_display', default=1000)
parser.add_argument('--val_iter', type=int, dest='val_iter', default=100)
config, unparsed = parser.parse_known_args() 


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.memory_usage)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

""" TRAINING """
x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
y_ = tf.placeholder(tf.float32, shape=[None, config.n_classes])
keep_prob = tf.placeholder(tf.float32)

print("\nLoding the model")

"""
Main Network
"""

slim = tf.contrib.slim
vgg = nets.vgg
res = nets.resnet_v1
inc = nets.inception

if config.model_name == 'vgg_19':

	with slim.arg_scope(vgg.vgg_arg_scope()):
		logits, endpoints = vgg.vgg_19(x, num_classes=config.n_classes, is_training=False)
		feat_layer = endpoints['vgg_19/fc7']
	all_vars = tf.all_variables()
	var_to_restore = [v for v in all_vars if not v.name.startswith('vgg_19/fc8')]


elif config.model_name == 'resnet_v1_50':
	res = nets.resnet_v1
	with slim.arg_scope(res.resnet_arg_scope()):
		logits, endpoints = res.resnet_v1_50(x, num_classes=config.n_classes, is_training=False)
		feat_layer = endpoints['resnet_v1_50/block4/unit_3/bottleneck_v1']
	all_vars = tf.all_variables()
	var_to_restore = [v for v in all_vars] # if not v.name.startswith('predictions')]


elif config.model_name == 'inception_v3':
	with slim.arg_scope(inc.inception_v3_arg_scope()):
		logits, endpoints = inc.inception_v3(x, num_classes=config.n_classes, is_training=False)
		feat_layer = endpoints['PreLogits']
	#all_vars = tf.all_variables()
	#var_to_restore = [v for v in all_vars] 

model = config.model_path + config.model_name + '.ckpt'

tf.train.start_queue_runners(sess=sess)
saver = tf.train.Saver(var_to_restore)
saver.restore(sess, model)

print("\nStart Extracting features")

for itr in range(100):
	print("\nFor the {0:03d}".format(itr))
awa_train_path = config.data_path + 'meta/path_label_list{0:03d}.txt'.format(itr)

# num_file : int 
# count the number of input image files
with open(awa_train_path) as f:
    for num_file, l in enumerate(f):
        pass

"""
example) queue_data('/home/siit/navi/data/sample/meta/path_label_list.txt', 
50, 1, 'val',multi_label=False)
"""
trainX, trainY = data_loader.queue_data(
		awa_train_path, config.n_classes, config.batch_size, 'val', multi_label=False)

feat = []
lab = []
path_feat = {}
for i in range(num_file+1):
	batch_x, batch_y = sess.run([trainX, trainY])
	_, idx = np.nonzero(batch_y)

	feature = sess.run(feat_layer, feed_dict={x: batch_x, y_: batch_y, keep_prob:1.0})
	feat.append(feature[0][0][0])

	lab.append(idx[0])
	path_feat[]
	
	if i%1000 == 0:
		print("{0:5f} % done".format(100*i/num_file))


save_path = config.save_path
if not os.path.exists(save_path):
	os.mkdir(save_path)

sio.savemat(save_path + config.model_name + '_feature_prediction{0:03d}.mat'.format(itr), 
	{'feature': feat, 'label': lab})

print('end')
