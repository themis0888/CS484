import tensorflow as tf
import os, random
import numpy as np
import pdb


class SISR:
	def __init__(self, sess, config, name):
		self.sess = sess
		self.name = name
		self._build_net(config)
		self.training = tf.placeholder(tf.bool)
		

	def _build_net(self, config):

		self.window = 3
		self.height = config.im_size
		self.width = config.im_size
		self.channels = 3
		self.n_classes = config.n_classes
		self.num_block = int(np.log(config.im_size/7)/np.log(2))
		self.layers = ['conv1', 
		'res1_conv1', 'res1_conv2', 'res2_conv1', 'res2_conv2', 
		'res3_conv1', 'res3_conv2', 'res4_conv1', 'res4_conv2', 
		'conv2', 'conv3', 'conv4']
		self.filt = {
			'conv1':64,
			'res1_conv1':64,
			'res1_conv2':64,
			'res2_conv1':64,
			'res2_conv2':64,
			'res3_conv1':64,
			'res3_conv2':64,
			'res4_conv1':64,
			'res4_conv2':64,
			'conv2':64,
			'conv3':256,
			'conv4':3
		}
		self.ratio = config.ratio
		self.im_size = [self.height, self.width, self.channels]
		self.n = 5
		self.reuse = False

		self.X = tf.placeholder(tf.float32, [None, self.width, self.height, self.channels])
		self.Y = tf.placeholder(tf.float32, 
		[None, self.ratio*self.width, self.ratio*self.height, self.channels])
		
		input_data = self.X
		input_data = tf.layers.conv2d(inputs=input_data, filters=64,
					kernel_size = [7,7], padding="same", activation=tf.nn.relu)
		
		for i in range(4):
			input_data = self.residual_block(input_data, 64, 3)
		
		input_data = tf.layers.conv2d(inputs=input_data, filters=64,
					kernel_size = [3,3], padding="same")

		input_data = tf.layers.conv2d(inputs=input_data, filters=256,
					kernel_size = [3,3], padding="same")
		
		input_data = tf.nn.relu(tf.depth_to_space(input_data, 2, 'NHWC'))

		self.output_data = tf.layers.conv2d(inputs=input_data, filters=3,
					kernel_size = [7,7], padding="same", activation=tf.nn.relu)

		# -------------------- Objective -------------------- #


		self.cost = tf.losses.absolute_difference(self.Y, self.output_data)
		total_var = tf.global_variables() 
		self.optimizer = tf.train.AdamOptimizer(0.001, epsilon=0.01).minimize(self.cost)
		
		init = tf.global_variables_initializer()
		self.sess.run(init)
		self.saver = tf.train.Saver(total_var)

		tf.train.start_queue_runners(sess=self.sess)

	def train(self, x_data, y_data, training=True):
		return self.sess.run([self.optimizer, self.cost], 
			feed_dict={self.X: x_data, self.Y: y_data, self.training: training})

	def test(self, x_test, training=False):
		return self.sess.run(self.output_data, 
			feed_dict={self.X: x_test, self.training: training})


	def residual_block(self, input_layer, num_filter, kernel):
		conv_layer = tf.layers.conv2d(inputs=input_layer, filters=self.filt[self.layers[0]],
					kernel_size = [7,7], padding="same", activation=tf.nn.relu)
		conv_layer = tf.layers.conv2d(inputs=conv_layer, filters=self.filt[self.layers[0]],
					kernel_size = [7,7], padding="same")
		return conv_layer + input_layer

	


