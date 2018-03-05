# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 16:17:00 2018

@author: sgage_000
"""

import tensorflow as tf
import numpy as np
import math
import mg_system_data
import matplotlib.pyplot as plt
tf.reset_default_graph()
#sess = tf.InteractiveSession()
from tensorflow.python.ops.control_flow_ops import cond


#Trying to make for general size, but only works for 6x6 gridsize

#Variables
level = 2
level_prime = 0
batch_size = 2
gridsize = 8
g1 = np.ceil(gridsize/2).astype(int)
g2 = np.ceil(g1/2).astype(int)
print ('g1/g2 = ',g1,'/',g2)
epoch_num = 1
num_batch = 1

def conv_restrict(input_tensor, level, level_prime):
    print ('restrict level {}'.format(level))
    size = int(gridsize/(2**level_prime))
    print ('size = {}'.format(size))
    a  = tf.gather(input_tensor, [size-1], axis=1)
    input_tensor = tf.concat([a,input_tensor], axis=1)
    input_tensor = tf.reshape(input_tensor, shape=[batch_size, size+1, 1])
    weights = tf.get_variable(name='R{}'.format(level),shape=[3,1,1]) 
    input_tensor = tf.expand_dims(input_tensor, 1)
    weights = tf.expand_dims(weights, 0) 
    print ('input_tensor = ', input_tensor, 'weights = ', weights)
    #conv = tf.reshape(tf.nn.conv2d(input_tensor, weights, strides=[1,1,2,1], padding="VALID"), [batch_size, size/2, 1])
    conv = tf.nn.conv2d(input_tensor, weights, strides=[1,1,2,1], padding="VALID")
    conv = tf.reshape(conv, [batch_size, int(size/2), 1])
    print ('conv = ', conv)
    return conv


def conv_prolong(input_tensor, level, level_prime):
    print ('prolong level {}'.format(level))
    size = int(gridsize/(2**level_prime))
    print ('size = {}'.format(size))
    weights = tf.get_variable(name = 'P{}'.format(level),shape= [3,1,1])
    input_tensor = tf.expand_dims(input_tensor, 1)
    weights = tf.expand_dims(weights, 0) #shape (1,g1+1,1`,1)
    print ('input_tensor = ', input_tensor, 'weights = ', weights)
    conv = tf.reshape(tf.nn.conv2d_transpose(input_tensor, weights, output_shape=[batch_size,1,size+1,1], strides=[1,1,2,1], padding="VALID"), [batch_size*(size+1),])
    #Deal with adding first padded element value to last element
    #conv = tf.placeholder("float", name='conv level{}'.format(level))
    conv_adjust = tf.Variable(conv, dtype=tf.float32, trainable=False, validate_shape=False, name="conv_adjust")
    #conv = tf.reshape(conv, [batch_size*(size+1), 1])
    #p = tf.gather(conv, [0],axis=1) #update values
    #print ('p = {}'.format(p))
    update_vals = np.array([])
    #for j in range(0,)
    indices = np.array([])
    for i in range(size, (size+1)*batch_size, size+1): #not efficient
        print ('i = {}'.format(i))
        indices = np.append(indices, [i])
        update_vals = np.append(update_vals, [i-size])
    indices = np.reshape(indices.astype(np.int32), [batch_size,]) #indices for update
    update_vals = np.reshape(update_vals.astype(np.int32),[batch_size,])
    conv_adjust = tf.scatter_add(conv_adjust, indices, update_vals)
    print ('indices = {}'.format(indices))
    print ('update values = {}'.format(update_vals))
    #last_el = conv[:,-1,:]
    #last_el += p
    #print ('last_el = {}'.format(last_el))
    #with tf.control_dependencies([conv[:,size-1,:].assign(last_el)]):
    conv_adjust = tf.reshape(conv_adjust, [batch_size, size+1, 1])
    print ('conv_adjust = {}'.format(conv_adjust))
    #tf.assign(conv[:,size-1,:], last_el)
    conv_adjust = conv_adjust[:, 1:,:]
    print ('conv_adjust = ', conv_adjust)
    return conv_adjust

def inverse_layer(input_tensor, level, level_prime):
    print ('inverse layer level {}'.format(level))
    size = int(gridsize/(2**level_prime))
    weights = tf.get_variable('Inverse_level{}'.format(level),shape=[batch_size,size, size])
    input_tensor = tf.reshape(input_tensor, [batch_size, 1, size])
    inv = tf.reshape(tf.matmul(input_tensor,weights), [batch_size, size, 1])
    print ('inv = {}'.format(inv))
    return inv

def diagonal(input_tensor, level, level_prime):
    print ('diagonal level {} '.format(level))
    size = int(gridsize/(2**(level_prime+1)))
    print ('size = ', size)
    D = tf.tile([0,1],[size],name='D{}'.format(level))
    D = tf.cast(D, tf.float32)
    c = tf.Variable(0.5,name='x')
    D = c*D
    D = tf.expand_dims(tf.expand_dims(D, 0), 0)
    print ('D = {}, input_tensor = {}'.format(D, tf.transpose(input_tensor)))
    Db = tf.reshape(D @ tf.transpose(input_tensor), [batch_size, 1 ,1])
    print ('Db = {}'.format(Db))
    return Db #this shape should be (1,1,8,1)

def model(b, level, level_prime):
    if level == 0:
        return inverse_layer(b, level, level_prime)
    else:
        if (b.shape[1] % 2 == 1):
            raise RuntimeError('Cannot restrict from level {} with shape {}'.format(level, b.shape[1]))
        print ('b.shape[1] = ', b.shape[1], b)
        Db = diagonal(b, level, level_prime)
        Rb = conv_restrict(b, level, level_prime)
        bc = model(Rb, level-1, level_prime+1)
        Mb = conv_prolong(bc, level, level_prime)
        return Db + Mb
    
b = tf.placeholder("float", shape=[None,gridsize,1], name='b')
u_ = tf.placeholder("float", shape=[None, gridsize,1], name='u')

output_u_ = model(b, level, level_prime)
output = tf.Print(output_u_, [output_u_])

loss = tf.losses.mean_squared_error(output_u_, u_)
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
            if (error <= 0.02):
                break
            loss_plt[e+shift] = error
            epoch[e+shift] = e+shift
        shift+=epoch_num  
	 # print ('output = ', sess.run(output, {b: np.array(b_vals), u_: np.array(actual_u_outputs)}))
	 # print ('b_vals = ', b_vals)
	 # print ('actual_u_outputs = ', actual_u_outputs)
	 # for var in tf.trainable_variables():
	 # 	print ('var name = ', var.name,' values = ', sess.run(var))
	 # print ('A inverse = ', A_inv)
    plt.plot(epoch,loss_plt)
    plt.title("Loss Plot")
    plt.xlabel("Cumulative Epoch")
    plt.ylabel("Mean squared error")
    plt.show()
		
network_training()
	    
