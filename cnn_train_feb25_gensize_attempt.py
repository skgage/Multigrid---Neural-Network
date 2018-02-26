# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 18:48:30 2018

@author: sgage_000
"""

import tensorflow as tf
import numpy as np
import math
import mg_system_data
import matplotlib.pyplot as plt


#Trying to make for general size, but only works for 6x6 gridsize
#2/26 11:22am is running for gridsize = 12, but the restriction is weird

#Variables
batch_size = 1
gridsize = 12
g1 = np.ceil(gridsize/2).astype(int)
g2 = np.ceil(g1/2).astype(int)
print ('g1/g2 = ',g1,'/',g2)
epoch_num = 1000
num_batch = 10

def conv_restrict(input_tensor, weights, biases, strides, padding, coarse_size):
    input_tensor = tf.expand_dims(input_tensor, 1)
    weights = tf.expand_dims(weights, 0) 
    print ('input_tensor = ', input_tensor, 'weights = ', weights)
    conv_temp = tf.nn.conv2d(input_tensor, weights, strides=strides, padding=padding)
    print ('conv_temp = ', conv_temp)
    conv = tf.reshape(tf.nn.conv2d(input_tensor, weights, strides=strides, padding=padding), [batch_size, coarse_size, 1])
    print ('conv = ', conv)
    pre_activation = tf.nn.bias_add(conv, biases)
    return tf.identity(pre_activation)


def conv_prolong(input_tensor, weights, biases, output_shape, strides, padding, coarse_size):
	input_tensor = tf.expand_dims(input_tensor, 1)
	weights = tf.expand_dims(weights, 0) #shape (1,g1+1,1`,1)
	print ('input_tensor = ', input_tensor, 'weights = ', weights)
	conv = tf.reshape(tf.nn.conv2d_transpose(input_tensor, weights, output_shape=output_shape, strides=strides, padding=padding), [batch_size, coarse_size, 1])
	print ('conv = ', conv)
	pre_activation = tf.nn.bias_add(conv, biases)
	return tf.identity(pre_activation)

def model(b, u_):
	with tf.name_scope('data'):
		a  = tf.gather(b, [gridsize-1], axis=1)
		b1 = tf.concat([a,b], axis=1)
		b1 = tf.reshape(b1, shape=[batch_size, gridsize+1, 1])	

	with tf.variable_scope('Restriction1') as scope:
		weights = tf.Variable(tf.random_normal([g1,1,1]),name='R2') 
		biases = tf.Variable(tf.zeros(1),name='b_R2')
		restrict_1 = conv_restrict(b1, weights, biases, [1,1,2,1], "VALID", g1-2) #IS THIS CORRECT STRIDE?
		
	with tf.variable_scope('Restriction2') as scope:
		weights = tf.Variable(tf.random_normal([g2-1,1,1]), name='R1')
		biases = tf.Variable(tf.zeros(1),name='b_R1')
		restrict_2 = tf.reshape(conv_restrict(restrict_1, weights, biases, [1,1,2,1], "SAME", g2-1), [batch_size, 1, g2-1]) #IS THIS CORRECT STRIDE?

	with tf.variable_scope('Inverse') as scope:
		weights3 = tf.Variable(tf.random_normal([batch_size, g2-1,g2-1]), name='A0inv')
		biases3 = tf.Variable(tf.zeros(g2-1),name='b_A0inv')
		h0 = tf.identity(tf.matmul(restrict_2, weights3)+biases3)
		h0 = tf.reshape(h0, [batch_size,g2-1,1])

	with tf.variable_scope('Prolongation1') as scope:
		weights = tf.Variable(tf.random_normal([g2+2,1,1]), name='P1')
		biases = tf.Variable(tf.zeros(1),name='b_P1')
		prolong_1 = conv_prolong(h0, weights, biases, [batch_size, 1, g1, 1], [1,1,1,1],"VALID", coarse_size=g1)

	#FOR NOW WILL DEFINE D1 AND D2 EXPLICTLY
	D1 = tf.constant([0,1,0,1,0,1],dtype=tf.float32)
	D1 = tf.reshape(D1, [1,g1,1])
	D2 = tf.constant([0,0.5,0,0.5,0,0.5,0,0.5,0,0.5,0,0.5], dtype=tf.float32)
	D2 = tf.reshape(D2, [1,gridsize,1])

	h1 = tf.add(prolong_1, D1)

	with tf.variable_scope('Prolongation2') as scope:
		weights = tf.Variable(tf.random_normal([g1+1,1,1]), name='P2')
		biases = tf.Variable(tf.zeros(1),name='b_P2')
		prolong_2 = conv_prolong(h1, weights, biases, [batch_size, 1, gridsize, 1], [1,1,1,1],"VALID", coarse_size=gridsize)
	
	h3 = tf.add(b, D2)
	output = tf.reshape(tf.add(h3, prolong_2), [batch_size, gridsize, 1])
	print ('u_ =', u_, 'output = ', output)
	return b, u_, output

b = tf.placeholder("float", shape=[None,gridsize,1], name='b')
u_ = tf.placeholder("float", shape=[None, gridsize,1], name='u')

original_b, original_u_, output_u_ = model(b, u_)
output = tf.Print(output_u_, [output_u_])

loss = tf.losses.mean_squared_error(output_u_, original_u_)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
# optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4)
train = optimizer.minimize(loss)

correct = tf.equal(output_u_, u_)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

def network_training():
	loss_plt = [0]*(epoch_num*num_batch)
	epoch = [0]*(epoch_num*num_batch)
	shift = 0
	for n in range(num_batch):
		b_vals, actual_u_outputs = mg_system_data.gen_data(gridsize, batch_size)
		for e in range(epoch_num):
		    sess.run(train, {b: b_vals.astype(np.float32), u_: actual_u_outputs.astype(np.float32)})
		   # print('epoch # ', e, 'training_performance =', sess.run(accuracy, {b: np.array(b_vals), u_: np.array(actual_u_outputs)}))
		    error = sess.run(loss, {b: np.array(b_vals), u_: np.array(actual_u_outputs)})
		    print('epoch # ', e+shift, 'training_error =', error)
		    loss_plt[e+shift] = error
		    epoch[e+shift] = e+shift
		shift+=epoch_num  
		# print ('output = ', sess.run(output, {b: np.array(b_vals), u_: np.array(actual_u_outputs)}))
		# print ('b_vals = ', b_vals)
		# print ('actual_u_outputs = ', actual_u_outputs)
		# for var in tf.trainable_variables():
		# 	print ('var name = ', var.name,' values = ', sess.run(var))
		#print ('A inverse = ', A_inv)
	plt.plot(epoch,loss_plt)
	plt.title("Loss Plot")
	plt.xlabel("Cumulative Epoch")
	plt.ylabel("Mean squared error")
	plt.show()
		
network_training()
	    
