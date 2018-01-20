'''

	Network Training for Convolutional Neural Networks of Multigrid Method
Network seeks to learn restriction and prolongation operators and a 
coarsened A inverse for 1D Poisson equation. Currently, gridsize is fixed
and the boundary conditions for the problem are considered periodic.
Later to be extended to two- and three-dimensional systems and varying
boundary conditions.

	Author:
	Sarah K Gage
	University of Colorado Boulder

	'''

import tensorflow as tf
import numpy as np
import math


import mg_system_data
A = mg_system_data.Laplacian(6)

batch_size = 1
gridsize = 6
train_b_sets, train_solution_sets = mg_system_data.gen_data(gridsize, 1) 
train_b_sets = train_b_sets.reshape((1, gridsize))
train_solution_sets = train_solution_sets.reshape((1, gridsize))
test_b_sets, test_solution_sets = mg_system_data.gen_data(gridsize, 1)
test_b_sets = test_b_sets.reshape((1, gridsize))
test_solution_sets = test_solution_sets.reshape((1, gridsize))

print (train_b_sets.shape, test_b_sets.shape)
print ('training b set = ', train_b_sets)
print ('training u set = ', train_solution_sets)
print ('A = ', A.shape)
print (np.matmul(train_solution_sets, A))
print (train_solution_sets.shape, test_solution_sets.shape)


def build_model(b):
	a  = [b[-1]]
	b1 = tf.concat([a,b], axis=0)
	b1 = tf.reshape(b1, shape=[1, 7, 1])
	weights = tf.Variable(tf.random_normal([3,1,1]),name='R2') 
	biases = tf.Variable(tf.zeros(1),name='b_R2')
	conv = tf.nn.conv1d(b1, weights, stride=2, padding="VALID")
	pre_activation = tf.nn.bias_add(conv, biases)
	conv1 = tf.identity(pre_activation)

	weights2 = tf.Variable(tf.random_normal([2,1,1]), name='R1')
	biases2 = tf.Variable(tf.zeros(1),name='b_R1')
	conv = tf.nn.conv1d(conv1, weights2, stride=2, padding="SAME")
	pre_activation = tf.nn.bias_add(conv, biases2)
	conv2 = tf.identity(pre_activation)
	conv2 = tf.reshape(conv2, [1,2])
	
	weights3 = tf.Variable(tf.random_normal([2,2]), name='A0inv')
	biases3 = tf.Variable(tf.zeros(2),name='b_A0inv')
	h0 = tf.identity(tf.matmul(conv2, weights3)+biases3)
	h0 = tf.reshape(h0, [1,2,1])

	weights4 = tf.Variable(tf.random_normal([1,1,2]), name='P1')
	biases4 = tf.Variable(tf.zeros(2),name='b_P1')
	conv = tf.nn.conv1d(h0, weights4, stride=1, padding="VALID")
	pre_activation = tf.nn.bias_add(conv, biases4)
	conv3 = tf.identity(pre_activation)
	h = tf.gather(conv3[:,:,1],[0], axis=1)+ tf.gather(conv3[:,:,1],[1], axis=1)
	h = tf.reshape(tf.concat([h, h],axis=0),[1,2])
	conv3 = (tf.reshape(tf.transpose(tf.stack([conv3[:,:,0], h])), (4,1)))
	conv3 = tf.gather(conv3, [0,1,2],axis=0)

	D1 = tf.constant([0,1,0],dtype=tf.float32)
	D1 = tf.reshape(D1, [1,3,1])
	D2 = tf.constant([0,0.5,0,0.5,0,0.5], dtype=tf.float32)
	D2 = tf.reshape(D2, [1,6,1])

	h1 = tf.add(conv3, D1)

	conv3 = tf.cast(tf.reshape(h1, [1,3,1]), dtype=tf.float32)
	weights5 = tf.Variable(tf.random_normal([1,1,3]), name='P2')
	biases5 = tf.Variable(tf.zeros(3),name='b_P2')
	conv = tf.nn.conv1d(h1, weights5, stride=1, padding="VALID")
	pre_activation = tf.nn.bias_add(conv, biases5)
	conv4 = tf.identity(pre_activation)
	h = conv4[:,:,2]+ tf.gather(conv4[:,:,0],[1,2,0], axis=1)
	conv4 = (tf.reshape(tf.transpose(tf.stack([conv4[:,:,1], h])), (6,1)))
	
	h3 = tf.add(b, D2)
	output = tf.add(h3, conv4)
	output = tf.reshape(output, [1,1,6])
	return output

def network_training(b_vals, actual_u_outputs, learning_rate, epoch_num, test_vals, test_outputs, process=False):
	b = tf.placeholder(tf.float32, shape=(None, 1, 6))
	u_ = tf.placeholder(tf.float32, shape=(None, 1, 6))

	output = build_model(b)

	loss = tf.losses.mean_squared_error(output, u_)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	train = optimizer.minimize(loss)

	correct = tf.equal(output, u_)
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)
	for e in range(epoch_num):
	    num_entries = len(b_vals)
	    for i in range(int(num_entries/batch_size)):
	    	print ('shape of desired output = ', actual_u_outputs.shape)
	    	sess.run(train, {b: np.array(b_vals), u_: np.array(actual_u_outputs)})
	    print('epoch # ', e, 'training_performance =', sess.run(accuracy, {b: np.array(b_vals), u_: np.array(actual_u_outputs)}))
	    print('epoch # ', e, 'training_error =', sess.run(loss, {b: np.array(b_vals), u_: np.array(actual_u_outputs)}))
	    if (process == True):
	     	test_num_entries = len(test_vals)
	     	#print('testing_performance', sess.run(accuracy, {x: np.array(test_vals), y_: np.array(test_outputs)}))
	print('training_performance after all epoch of trial = ', sess.run(accuracy, {b: np.array(b_vals), u_: np.array(actual_u_outputs)}))
	#print('testing_performance after all epoch of trial = ', sess.run(accuracy, {x: np.array(test_vals), y_: np.array(test_outputs)}))
	print ('output = ', sess.run(output, {b: np.array(b_vals), u_: np.array(actual_u_outputs)}))
	print ('actual values = ', actual_u_outputs)

network_training(train_b_sets, train_solution_sets, 0.001, 1000, test_b_sets, test_solution_sets)
	    
