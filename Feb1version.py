import tensorflow as tf
import numpy as np
import math
import mg_system_data
# import cnn_model


#FOR NOW A IS ALWAYS 6X6 AND PERIODIC LAPLACIAN

A = mg_system_data.Laplacian(6)
A_inv = np.linalg.pinv(A)
#8:33am 12/16 changed reshape below to 2d
#import mg_model
batch_size = 1
gridsize = 6
train_b_sets, train_solution_sets = mg_system_data.gen_data(gridsize, 1) 
train_b_sets = train_b_sets.reshape((1,gridsize, 1))#.astype(dtype=np.float32)
train_solution_sets = train_solution_sets.reshape((1,gridsize, 1))#.astype(dtype=np.float32)
#test_b_sets, test_solution_sets = mg_system_data.gen_data(gridsize, 1)
#test_b_sets = test_b_sets.reshape((1, 1, gridsize))
#test_solution_sets = test_solution_sets.reshape((1, 1, gridsize))

# print (train_b_sets.shape, test_b_sets.shape)
print ('training b set = ', train_b_sets.shape, train_b_sets.dtype)
print ('training u set = ', train_solution_sets.shape, train_solution_sets.dtype)
print ('A = ', A.shape)
#print (np.matmul(train_solution_sets, A))
# print (train_solution_sets.shape, test_solution_sets.shape)

def model(b, u_):
	with tf.name_scope('data'):
		# b = tf.placeholder("float", shape=[1,6,1], name='b')
		# u_ = tf.placeholder("float", shape=[1, 6,1], name='u')
		#b = tf.reshape(b, [6,1])
		#u_ = tf.reshape(u_, [6,1])
		a  = tf.gather(b, [5], axis=1)
		b1 = tf.concat([a,b], axis=1)
		b1 = tf.reshape(b1, shape=[1, 7, 1])	

	with tf.variable_scope('Restriction1') as scope:
		weights = tf.Variable(tf.random_normal([3,1,1]),name='R2') 
		biases = tf.Variable(tf.zeros(1),name='b_R2')
		conv = tf.nn.conv1d(b1, weights, stride=2, padding="VALID")
		pre_activation = tf.nn.bias_add(conv, biases)
		conv1 = tf.identity(pre_activation)
# print ('size = ',(tf.size(conv1)).eval())
# print ('shape = ', tf.shape(conv1).eval())
	with tf.variable_scope('Restriction2') as scope:
		weights2 = tf.Variable(tf.random_normal([2,1,1]), name='R1')
		biases2 = tf.Variable(tf.zeros(1),name='b_R1')
		conv = tf.nn.conv1d(conv1, weights2, stride=2, padding="SAME")
		pre_activation = tf.nn.bias_add(conv, biases2)
		conv2 = tf.identity(pre_activation)
		conv2 = tf.reshape(conv2, [1,2])
# print ('size = ',(tf.size(conv2)).eval())
# print ('shape = ', tf.shape(conv2).eval())
	with tf.variable_scope('Inverse') as scope:
		weights3 = tf.Variable(tf.random_normal([2,2]), name='A0inv')
		biases3 = tf.Variable(tf.zeros(2),name='b_A0inv')
		h0 = tf.identity(tf.matmul(conv2, weights3)+biases3)
		h0 = tf.reshape(h0, [1,2,1])
# print ('size = ',(tf.size(h0)).eval())
# print ('shape = ', tf.shape(h0).eval())
	with tf.variable_scope('Prolongation1') as scope:
		weights4 = tf.Variable(tf.random_normal([1,1,2]), name='P1')
		biases4 = tf.Variable(tf.zeros(2),name='b_P1')
		conv = tf.nn.conv1d(h0, weights4, stride=1, padding="VALID")
		pre_activation = tf.nn.bias_add(conv, biases4)
		conv3 = tf.identity(pre_activation)
	h = tf.gather(conv3[:,:,1],[0], axis=1)+ tf.gather(conv3[:,:,1],[1], axis=1)
	h = tf.reshape(tf.concat([h, h],axis=0),[1,2])
	conv3 = (tf.reshape(tf.transpose(tf.stack([conv3[:,:,0], h])), (4,1)))
	conv3 = tf.gather(conv3, [0,1,2],axis=0)
	#FOR NOW WILL DEFINE D1 AND D2 EXPLICTLY
	D1 = tf.constant([0,1,0],dtype=tf.float32)
	D1 = tf.reshape(D1, [1,3,1])
	D2 = tf.constant([0,0.5,0,0.5,0,0.5], dtype=tf.float32)
	D2 = tf.reshape(D2, [1,6,1])
#h1 = conv3 [1,1,3] + D1 [1,1,3]
	h1 = tf.add(conv3, D1)
	#this is more P2!
	with tf.variable_scope('Prolongation2') as scope:
		conv3 = tf.cast(tf.reshape(h1, [1,3,1]), dtype=tf.float32)
		weights5 = tf.Variable(tf.random_normal([1,1,3]), name='P2')
		biases5 = tf.Variable(tf.zeros(3),name='b_P2')
		conv = tf.nn.conv1d(h1, weights5, stride=1, padding="VALID")
		pre_activation = tf.nn.bias_add(conv, biases5)
		conv4 = tf.identity(pre_activation)
	h = conv4[:,:,2]+ tf.gather(conv4[:,:,0],[1,2,0], axis=1)
	conv4 = (tf.reshape(tf.transpose(tf.stack([conv4[:,:,1], h])), (6,1)))

	b = tf.reshape(b, [1,6,1])
	h3 = tf.add(b, D2)
	output = tf.add(h3, conv4)
	output = tf.reshape(output, [1,6,1])
# print ('size output= ',(tf.size(output)).eval())
# print ('shape = ', tf.shape(output).eval())
#return output
	return b, u_, output

#print (output.eval())
#tf.reset_default_graph()
b = tf.placeholder("float", shape=[1,6,1], name='b')
u_ = tf.placeholder("float", shape=[1, 6,1], name='u')

original_b, original_u_, output_u_ = model(b, u_)
output = tf.Print(output_u_, [output_u_])

loss = tf.losses.absolute_difference(output_u_, u_)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4)
# optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4)
train = optimizer.minimize(loss)

correct = tf.equal(output_u_, u_)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

def network_training(b_vals, actual_u_outputs, epoch_num):
	#tf.reset_default_graph()
	for e in range(epoch_num):
	    #print(e)
	    # num_entries = len(b_vals)
	    #print ('shape b_vals = ', b_vals.shape)
	    # for i in range(int(num_entries/batch_size)):
	    # print ('shape of b output = ', b_vals.astype(np.float32).shape)
	    sess.run(train, {b: b_vals.astype(np.float32), u_: actual_u_outputs.astype(np.float32)})
	    print('epoch # ', e, 'training_performance =', sess.run(accuracy, {b: np.array(b_vals), u_: np.array(actual_u_outputs)}))
	    print('epoch # ', e, 'training_error =', sess.run(loss, {b: np.array(b_vals), u_: np.array(actual_u_outputs)}))
	#     print ('output of model after 1 epoch = ', sess.eval(output, {b: np.array(b_vals)}))
	#     if (process == True):
	#      	test_num_entries = len(test_vals)
	#      	#print('testing_performance', sess.run(accuracy, {x: np.array(test_vals), y_: np.array(test_outputs)}))
	# #print ('weights ', weights)
	# print('training_performance after all epoch of trial = ', sess.run(accuracy, {b: np.array(b_vals), u_: np.array(actual_u_outputs)}))
	# #print('testing_performance after all epoch of trial = ', sess.run(accuracy, {x: np.array(test_vals), y_: np.array(test_outputs)}))
	print ('output = ', sess.run(output, {b: np.array(b_vals), u_: np.array(actual_u_outputs)}))
	print ('b_vals = ', b_vals)
	print ('actual_u_outputs = ', actual_u_outputs)
	for var in tf.trainable_variables():
		print ('var name = ', var.name,' values = ', sess.run(var))
	#print (sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))
	print ('A inverse = ', A_inv)
network_training(train_b_sets, train_solution_sets, 1000)
	    
