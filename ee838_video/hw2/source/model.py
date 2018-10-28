import tensorflow as tf
import os, random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage.io as skio
import module as layer
import module as L
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
		self.lr = config.lr
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

		# take the input data and target data in flexible size 
		self.X = tf.placeholder(tf.float32, [None, None, None, self.channels])
		self.Y = tf.placeholder(tf.float32, 
			[None, None, None, self.channels])

		stg = []	
		# Stage 1
		net = self.X


		stg = []	
		X = tf.placeholder(tf.float32, [None, None, None, 3])
		net = X
		net = L.conv(net, name="conv1_1", kh=3, kw=3, n_out=64)
		net = L.conv(net, name="conv1_2", kh=3, kw=3, n_out=64)
		stg.append(net)
		net = L.pool(net, 'pool_1', 2, 2, 2, 2)

		net = L.conv(net, name="conv2_1", kh=3, kw=3, n_out=128)
		net = L.conv(net, name="conv2_2", kh=3, kw=3, n_out=128)
		stg.append(net)
		net = L.pool(net, 'pool_2', 2, 2, 2, 2)

		net = L.conv(net, name="conv3_1", kh=3, kw=3, n_out=256)
		net = L.conv(net, name="conv3_2", kh=3, kw=3, n_out=256)
		net = L.conv(net, name="conv3_3", kh=3, kw=3, n_out=256)
		stg.append(net)
		net = L.pool(net, 'pool_3', 2, 2, 2, 2)

		net = L.conv(net, name="conv4_1", kh=3, kw=3, n_out=512)
		net = L.conv(net, name="conv4_2", kh=3, kw=3, n_out=512)
		net = L.conv(net, name="conv4_3", kh=3, kw=3, n_out=512)
		stg.append(net)
		net = L.pool(net, 'pool_4', 2, 2, 2, 2)

		net = L.conv(net, name="conv5_1", kh=3, kw=3, n_out=512)
		net = L.conv(net, name="conv5_2", kh=3, kw=3, n_out=512)
		net = L.conv(net, name="conv5_3", kh=3, kw=3, n_out=512)
		stg.append(net)
		net = L.pool(net, 'pool_5', 2, 2, 2, 2)

		net = L.conv(net, name="conv6_1", kh=3, kw=3, n_out=512)
		net = L.conv(net, name="conv6_2", kh=3, kw=3, n_out=512)

		net = L.deconv(net, name="deconv5_1", kh=3, kw=3, n_out=512)
		net = tf.concat([net, stg[4]], axis=3)
		net = L.conv(net, name="deconv5_2", kh=3, kw=3, n_out=512)

		net = L.deconv(net, name="deconv4_1", kh=3, kw=3, n_out=512)
		net = tf.concat([net, stg[3]], axis=3)
		net = L.conv(net, name="deconv4_2", kh=3, kw=3, n_out=512)

		net = L.deconv(net, name="deconv3_1", kh=3, kw=3, n_out=256)
		net = tf.concat([net, stg[2]], axis=3)
		net = L.conv(net, name="deconv3_2", kh=3, kw=3, n_out=256)

		net = L.deconv(net, name="deconv2_1", kh=3, kw=3, n_out=128)
		net = tf.concat([net, stg[1]], axis=3)
		net = L.conv(net, name="deconv2_2", kh=3, kw=3, n_out=128)

		net = L.deconv(net, name="deconv1_1", kh=3, kw=3, n_out=64)
		net = tf.concat([net, stg[0]], axis=3)
		net = L.conv(net, name="deconv1_2", kh=3, kw=3, n_out=64)

		net = L.conv(net, name="deconv0_1", kh=3, kw=3, n_out=3)
		net = tf.concat([net, X], axis=3)
		net = L.conv(net, name="deconv0_2", kh=3, kw=3, n_out=3)

		self.output_data = net

		# -------------------- Objective -------------------- #

		# L1 loss 
		self.tau = 0.95
		alpha = tf.maximum(self.X - tf.ones(self.X.shape)*self.tau, tf.zeros(self.X.shape))
		self.inv = tf.pow(tf.divide(0.6*self.X, tf.maximum(1.6-self.X, 1e-10)), 1.0/0.9)
		self.hdr_img = (1-alpha)*self.inv + alpha*self.exp(self.output_data)
		self.cost = tf.losses.absolute_difference(self.Y, self.hdr_img)
		self.var_to_restore = tf.global_variables()
		self.optimizer = tf.train.AdamOptimizer(self.lr, epsilon=0.0005).minimize(self.cost)
		
		self.total_var = tf.global_variables() 
		init = tf.global_variables_initializer()
		self.sess.run(init)
		
		# restore the checkpoint if testing, or continue the learning
		if config.mode == 'testing' or config.re_train:
			self.saver = tf.train.Saver(self.var_to_restore)
			self.saver.restore(self.sess, tf.train.latest_checkpoint(config.checkpoint_path))
		self.saver = tf.train.Saver(self.total_var)
		
		tf.train.start_queue_runners(sess=self.sess)

	# the structure of the residual block 
	def residual_block(self, input_layer, num_filter, kernel):
		conv_layer = tf.layers.conv2d(inputs=input_layer, filters=self.filt[self.layers[0]],
					kernel_size = [7,7], padding="same", activation=tf.nn.relu)
		conv_layer = tf.layers.conv2d(inputs=conv_layer, filters=self.filt[self.layers[0]],
					kernel_size = [7,7], padding="same")
		return conv_layer + input_layer

	def train(self, x_data, y_data, training=True):
		return self.sess.run([self.optimizer, self.cost], 
			feed_dict={self.X: x_data, self.Y: y_data, self.training: training})

	def test(self, x_test, training=False):
		return self.sess.run(self.output_data, 
			feed_dict={self.X: x_test, self.training: training})

	# visualizing the images. You can find this from smaple_path directory
	def visualize(self, input_files, target_files, sample_path, counter, save_output = False, args = None):

		num_input = min(3, len(input_files))
		num_col = 3
		fig=plt.figure(figsize=(16, 16))

		output_files = self.sess.run(self.output_data, 
			feed_dict={self.X: input_files, self.Y: target_files, self.training: True})
		output_files = np.minimum(1, np.maximum(0, output_files))

		for i in range(num_input):

			input_file = input_files[i]
			output_file = output_files[i]
			target_file = target_files[i]
			#pdb.set_trace()

			fig.add_subplot(num_input, num_col, num_col*i+1)
			plt.imshow(input_file)
			fig.add_subplot(num_input, num_col, num_col*i+2)
			plt.imshow(output_file)
			fig.add_subplot(num_input, num_col, num_col*i+3)
			plt.imshow(target_file)

		plt.savefig(os.path.join(sample_path, '{0:06d}k_step.jpg'.format(counter)))
		plt.close()
		
		if save_output:
			skio.imsave(os.path.join(sample_path,'{:03d}.jpg'.format(counter)), output_file)

		return output_files



	


