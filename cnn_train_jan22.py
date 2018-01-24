import tensorflow as tf
import numpy as np
import math


import mg_system_data
import cnn_model


#FOR NOW A IS ALWAYS 6X6 AND PERIODIC LAPLACIAN

A = mg_system_data.Laplacian(6)
#8:33am 12/16 changed reshape below to 2d
#import mg_model
batch_size = 1
gridsize = 6
train_b_sets, train_solution_sets = mg_system_data.gen_data(gridsize, 1) 
train_b_sets = train_b_sets.reshape((1,gridsize, 1))
train_solution_sets = train_solution_sets.reshape((1,gridsize, 1))
test_b_sets, test_solution_sets = mg_system_data.gen_data(gridsize, 1)
#test_b_sets = test_b_sets.reshape((1, 1, gridsize))
#test_solution_sets = test_solution_sets.reshape((1, 1, gridsize))

print (train_b_sets.shape, test_b_sets.shape)
print ('training b set = ', train_b_sets)
print ('training u set = ', train_solution_sets)
print ('A = ', A.shape)
#print (np.matmul(train_solution_sets, A))
print (train_solution_sets.shape, test_solution_sets.shape)


b, u_, output = cnn_model.model(6)
output = tf.Print(output, [output])

loss = tf.losses.absolute_difference(output, u_)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4)
optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4)
train = optimizer.minimize(loss)

correct = tf.equal(output, u_)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

sess = tf.Session()


def network_training(b_vals, actual_u_outputs, epoch_num):
	for e in range(epoch_num):
	    #print(e)
	    num_entries = len(b_vals)
	    #print ('shape b_vals = ', b_vals.shape)
	    for i in range(int(num_entries/batch_size)):
	    	print ('shape of desired output = ', actual_u_outputs.shape)
	    	sess.run(train, {b: np.array(b_vals), u_: np.array(actual_u_outputs)})
	    print('epoch # ', e, 'training_performance =', sess.run(accuracy, {b: np.array(b_vals), u_: np.array(actual_u_outputs)}))
	    print('epoch # ', e, 'training_error =', sess.run(loss, {b: np.array(b_vals), u_: np.array(actual_u_outputs)}))
	    print ('output of model after 1 epoch = ', sess.eval(output, {b: np.array(b_vals)}))
	    if (process == True):
	     	test_num_entries = len(test_vals)
	     	#print('testing_performance', sess.run(accuracy, {x: np.array(test_vals), y_: np.array(test_outputs)}))
	#print ('weights ', weights)
	print('training_performance after all epoch of trial = ', sess.run(accuracy, {b: np.array(b_vals), u_: np.array(actual_u_outputs)}))
	#print('testing_performance after all epoch of trial = ', sess.run(accuracy, {x: np.array(test_vals), y_: np.array(test_outputs)}))
	print ('output = ', sess.run(output, {b: np.array(b_vals), u_: np.array(actual_u_outputs)}))
	print ('actual values = ', actual_u_outputs)

network_training(train_b_sets, train_solution_sets, 1)
	    
