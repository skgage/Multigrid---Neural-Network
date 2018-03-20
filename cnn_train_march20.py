#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 15:15:56 2018

@author: sarahgage
"""


import tensorflow as tf
import numpy as np
import math
import mg_system_data
import matplotlib.pyplot as plt
tf.reset_default_graph()
#sess = tf.InteractiveSession()
from tensorflow.python.ops.control_flow_ops import cond
import xlwt


#Variables
level = 3
level_prime = 0
batch_size = 1
#batch_size = tf.placeholder(dtype=tf.int32, name="batch_size")
num_data = 200
gridsize = 8
epoch_num = 1000
num_batch = 1
dim = 1 #number of dimensions 
mean = np.reshape([0.5, 1, 0.5], [1,3,1,1])
sigma = 0.0

def conv_restrict(input_tensor, level, level_prime):
    batch_size = tf.shape(input_tensor)[0]
    #print ('restrict level {}'.format(level))
    size = int(gridsize/(2**level_prime))
    #print ('size = {}'.format(size))
    a  = tf.gather(input_tensor, [size-1], axis=2)
    input_tensor = tf.concat([a,input_tensor], axis=2)
    input_tensor = tf.reshape(input_tensor, shape=[batch_size,1, size+1, 1])
    init = tf.constant(np.random.normal(mean, sigma, size=(1,3,1,1)),dtype=tf.float32)
    #weights = tf.get_variable(name='R{}'.format(level),shape=[1,3,1,1], initializer=init) 
    weights = tf.get_variable(name='R{}'.format(level),initializer=init, trainable=True) 
    #input_tensor = tf.expand_dims(input_tensor, 1)
    #weights = tf.expand_dims(weights, 0) 
    #print ('input_tensor = ', input_tensor, 'weights = ', weights)
    #conv = tf.reshape(tf.nn.conv2d(input_tensor, weights, strides=[1,1,2,1], padding="VALID"), [batch_size, size/2, 1])
    conv = tf.nn.conv2d(input_tensor, weights, strides=[1,1,2,1], padding="VALID")
    conv = tf.reshape(conv, [batch_size,1, int(size/2), 1])
    #print ('conv = ', conv)
    return conv


def conv_prolong(input_tensor, level, level_prime):
    batch_size = tf.shape(input_tensor)[0]
    #print ('prolong level {}'.format(level))
    size = int(gridsize/(2**level_prime))
    #print ('size = {}'.format(size))
    init = tf.constant(np.random.normal(mean, sigma, size=(1,3,1,1)),dtype=tf.float32)
    #weights = tf.get_variable(name = 'P{}'.format(level),shape= [1,3,1,1], initializer=init)
    weights = tf.get_variable(name = 'P{}'.format(level), initializer=init, trainable=True)   
    #input_tensor = tf.expand_dims(input_tensor, 1)
    #weights = tf.expand_dims(weights, 0) #shape (1,g1+1,1`,1)
    #print ('input_tensor = ', input_tensor, 'weights = ', weights)
    conv = tf.nn.conv2d_transpose(input_tensor, weights, output_shape=[batch_size,1,size+1,1], strides=[1,1,2,1], padding="VALID")
    iconv = conv[:,:,1:,:] + tf.pad(conv[:,:,0:1,:], [(0,0), (0,0), (size-1,0), (0,0)])
    return tf.reshape(iconv, (batch_size,1,size,1))

def inverse_layer(input_tensor, level, level_prime):
    #batch = tf.shape(input_tensor)[0]
    #batch_size = batch
    #if batch == num_data:
    #   batch_size1 = num_data
    #print ('inverse layer level {}'.format(level))
    size = int(gridsize/(2**level_prime))
    #weights = tf.Variable(tf.random_normal((tf.shape(input_tensor)[0],size, size)), validate_shape=False, name = 'Inverse_level{}'.format(level))
    weights = tf.get_variable(name='Inverse_level{}'.format(level), shape=[size, size], validate_shape=False) 
    weights = tf.expand_dims(tf.expand_dims(tf.ones([tf.shape(input_tensor)[0],1]), 1),1) * weights
    input_tensor = tf.reshape(input_tensor, [tf.shape(input_tensor)[0],1, 1, size])
    #input_tensor = tf.squeeze(input_tensor)
    inv = tf.reshape(tf.matmul(input_tensor,weights), [tf.shape(input_tensor)[0], 1,size, 1])
    #print ('inv = {}'.format(inv))
    return inv

def diagonal(input_tensor, level, level_prime):
    batch_size = tf.shape(input_tensor)[0]
    #print ('diagonal level {} '.format(level))
    size = int(gridsize/(2**(level_prime+1)))
    #print ('size = ', size)
    D = tf.tile([0,1],[size],name='D{}'.format(level))
    D = tf.cast(D, tf.float32)
    c = tf.Variable(0.5,name='x{}'.format(level), trainable=True)
    D = c*D
    D = tf.expand_dims(tf.expand_dims(tf.expand_dims(D, 0), 0),0)
    #print ('D = {}, input_tensor = {}'.format(D, tf.transpose(input_tensor)))
    Db = tf.reshape(D @ tf.reshape(tf.transpose(input_tensor),[1,1,size*2,batch_size]), [batch_size,1, 1 ,1])
    #print ('Db = {}'.format(Db))
    return Db #this shape should be (1,1,8,1)

def model(b, level, level_prime):
    if level == 0:
        return inverse_layer(b, level, level_prime)
    else:
        if (b.shape[2] % 2 == 1):
            raise RuntimeError('Cannot restrict from level {} with shape {}'.format(level, b.shape[1]))
        Db = diagonal(b, level, level_prime)
        Rb = conv_restrict(b, level, level_prime)
        bc = model(Rb, level-1, level_prime+1)
        Mb = conv_prolong(bc, level, level_prime)
        return Db + Mb
    
b = tf.placeholder("float", shape=[None,1,gridsize,1], name='b')
u_ = tf.placeholder("float", shape=[None,1, gridsize,1], name='u')

output_u_ = model(b, level, level_prime)
output = tf.Print(output_u_, [output_u_])

loss = tf.losses.mean_squared_error(output_u_, u_)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
train = optimizer.minimize(loss)

correct = tf.equal(output_u_, u_)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))



def network_training():
    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)
    loss_plt = [0]*(epoch_num*num_batch)
    epoch = [0]*(epoch_num*num_batch)
    shift = 0
    for n in range(num_batch):
        b_vals, actual_u_outputs = mg_system_data.gen_data(gridsize, num_data, dim)
        init_vars = {}
        i=1
        for var in tf.trainable_variables():
            var_val = sess.run(var)
            init_vars['{}'.format(var.name)]= var_val
            i+=1
        for e in range(epoch_num):
            for i in range(int(num_data/batch_size)):
                batch_x = b_vals[i:i+batch_size,:,:]
                batch_y = actual_u_outputs[i:i+batch_size,:,:]
                sess.run(train, {b: batch_x.astype(np.float32), u_: batch_y.astype(np.float32)})
            error = sess.run(loss, {b: np.array(b_vals), u_: np.array(actual_u_outputs)})
            print('epoch # ', e+shift, 'training_error =', error)
            loss_plt[e+shift] = error
            epoch[e+shift] = e+shift
        converged_vars = {}
        #converged_u = sess.run(output_u_, {b: np.array(b_vals), u_: np.array(actual_u_outputs)})
        j=i+1
        for var in tf.trainable_variables():
            var_val = sess.run(var)
            #sh.write(trial+1,j,'{}'.format(var_val))
            #print ('var name = ', var.name,' values = ', var_val)
            converged_vars['{}'.format(var.name)]= var_val
            j+=1
        print ('initial params = {}'.format(init_vars))
        print ('converged params = {}'.format(converged_vars))
        print('epoch # ', e+shift, 'training_error =', error)
    plt.plot(epoch,loss_plt)
    plt.title("Loss Plot")
    plt.xlabel("Cumulative Epoch")
    plt.ylabel("Mean squared error")
    plt.show()
    return error,converged_vars, init_vars, epoch, loss_plt 


network_training()
