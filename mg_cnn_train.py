import tensorflow as tf
import numpy as np
import math


import mg_system_data
#import mg_model
batch_size = 1
gridsize = 2
train_input_sets, train_solution_sets = mg_system_data.gen_data(gridsize, 1000) 
train_input_sets = train_input_sets.reshape((1000*gridsize, gridsize+1))
train_solution_sets = train_solution_sets.reshape((1000*gridsize, 1))
test_input_sets, test_solution_sets = mg_system_data.gen_data(gridsize, 100)
test_input_sets = test_input_sets.reshape((100*gridsize, gridsize+1))
test_solution_sets = test_solution_sets.reshape((100*gridsize, 1))

print (train_input_sets.shape, test_input_sets.shape)
print ('length of training set = ', train_input_sets[0:1])
print (train_solution_sets.shape, test_solution_sets.shape)


def build_(b): 

	def P(n, stencil):
	    stencil = stencil.eval()   
		m = int(np.ceil(n/2))
		print ('m =', m)
		A = np.zeros([n,n])
		i = 0
		j1 = 0
		j2 = 2
		for k in range(n):
			if k % 2 == 0:
				A[k,i]=stencil[0]
				i = i +1
			else:
				A[k,j1:j2]= stencil[1:]
				j1= j1 + 1
				j2 = j2+1
		A = np.delete(A, np.s_[m:], 1)
	    #A[::2] = stencil[0]
		return tf.convert_to_tensor(A)



	size2 = len(b)
	size1 = math.ceil(size2/2)
	size0 = math.ceil(size1/2)



	with tf.variable_scope('fully_connected1') as scope:		
		w0 = tf.get_variable("w0", [gridsize + 1, num_hidden_units_1])
		b0 = tf.Variable(tf.zeros(num_hidden_units_1))
		h0 = tf.identity(tf.matmul(x, w0)+b0, name="Hidden1")
		print (h0)

	weights = { #1st fix shapes and then write out operation order
	#how to require R and P to sum to 1?
	#okay to just make R the transpose of P?
		'row': tf.get_variable([1,1,3])
		'P2': tf.Variable(P(size2, row)),
		'R2': tf.Variable(tf.transpose(weights['P2'])),

		'P1': tf.Variable(P(size1, row)),
		'R1': tf.Variable(tf.transpose(weights['P1'])),

		'A0inv': tf.get_variable([size0, size0])
		
		}
	biases = { #are biases for P and R needed? not training
		'b_row': tf.Variable(tf.zeros(1))

		'b_P2': tf.Variable(tf.zeros(size2)),
		'b_R2': tf.Variable(tf.zeros(num_hidden_units_1)),

		'b_P1': tf.Variable(tf.zeros(num_hidden_units_1)),
		'b_R1': tf.Variable(tf.zeros(num_hidden_units_1)),

		'b_A0inv': tf.Variable(tf.zeros(size0))
	}
	#reshape input to a 3d tensor, [batch_size, input len, input dim]
	b = tf.reshape(b, shape=[batch, 1, gridsize])

	conv = tf.nn.conv1d(b, weights['row'], stride=1, padding='same')
	pre_activation = tf.nn.bias_add(conv, biases['b_row'])
    conv1 = tf.identity(pre_activation, name='r')

	expand = tf.Variable(tf.transpose(P(size2, conv1)), name='R2')

	
	conv2 = tf.nn.conv1d(weights['R2'],  )



	weights = tf.get_variable(name='A0inv', [size0,size0])
	#where does conv1 come in? am i using 'row' correctly?


	#WAY 2!!!!!!! not sure it this way is actually accomplishing learning the sparsity structure of b
	inner = tf.identity(tf.matmul(weights['P1'],tf.matmul(weights['A0inv'],weights['R1'])+biases['b_A0inv']))
	inner = inner + D1 #how to get this?

	outer1 = tf.identity(tf.matmul(tf.matmul(weights['P2'],inner),tf.matmul(weights['R2'],b)))

	outer = outer1 + tf.matmul(D2, b) #does this work? #how to get D2

	output = outer

	return output

	#reshape input to a 3d tensor, [batch_size, input len, input dim]
	A = tf.reshape(A, shape=[batch, gridsize, gridsize])
	#Convolutional layer
	conv1 = tf.nn.conv1d(A, weights['P']+biases['b_P'], stride=1, padding='same')

	conv2 = tf.nn.conv1d(conv1, weights['R']+biases['b_R'], stride=1, padding='same')

	output = tf.matmul(tf.matmul(conv2, A), conv1)

	return output

def build_model(b):
	b = tf.reshape(b, shape=[1, 1, 6])
	weights = tf.Variable(tf.random_normal([1,6,3]),name='R2')
	biases = tf.Variable(tf.zeros(3),name='b_R2')
	conv = tf.nn.conv1d(b, weights, stride=1, padding="SAME")
	pre_activation = tf.nn.bias_add(conv, biases)
	conv1 = tf.identity(pre_activation)
	print ('size = ',(tf.size(conv1)).eval())
	print ('shape = ', tf.shape(conv1).eval())

	weights2 = tf.Variable(tf.random_normal([1,3,2]), name='R1')
	biases2 = tf.Variable(tf.zeros(2),name='b_R1')
	conv = tf.nn.conv1d(conv1, weights2, stride=1, padding="SAME")
	pre_activation = tf.nn.bias_add(conv, biases2)
	conv2 = tf.identity(pre_activation)
	print ('size = ',(tf.size(conv2)).eval())
	print ('shape = ', tf.shape(conv2).eval())

	weights3 = tf.Variable(tf.random_normal([1,2,2]), name='A0inv')
	biases3 = tf.Variable(tf.zeros(2),name='b_A0inv')
	h0 = tf.identity(tf.matmul(conv2, weights3)+biases3)
	print ('size = ',(tf.size(h0)).eval())
	print ('shape = ', tf.shape(h0).eval())

	weights4 = tf.Variable(tf.random_normal([1,2,3]), name='P1')
	biases4 = tf.Variable(tf.zeros(3),name='b_P1')
	conv = tf.nn.conv1d(h0, weights4, stride=1, padding="SAME")
	pre_activation = tf.nn.bias_add(conv, biases4)
	conv3 = tf.identity(pre_activation)
	print ('size = ',(tf.size(conv3)).eval())
	print ('shape = ', tf.shape(conv3).eval())

	#FOR NOW WILL DEFINE D1 AND D2 EXPLICTLY
	D1 = tf.constant([0,1,0],dtype=tf.float32)
	D1 = tf.reshape(D1, [1,1,3])
	D2 = tf.constant([0,0.5,0,0.5,0,0.5], dtype=tf.float32)
	D2 = tf.reshape(D2, [1,1,6])
	#h1 = conv3 [1,1,3] + D1 [1,1,3]
	h1 = conv3 + D1
	print ('size = ',(tf.size(h1)).eval())
	print ('shape = ', tf.shape(h1).eval())

	weights5 = tf.Variable(tf.random_normal([1,3,6]), name='P2')
	biases5 = tf.Variable(tf.zeros(6),name='b_P2')
	conv = tf.nn.conv1d(conv3, weights5, stride=1, padding="SAME")
	pre_activation = tf.nn.bias_add(conv, biases5)
	conv4 = tf.identity(pre_activation)
	print ('size = ',(tf.size(conv4)).eval())
	print ('shape = ', tf.shape(conv4).eval())

	h3 = b + D2
	output = h3 + conv4
	print ('size output= ',(tf.size(output)).eval())
	print ('shape = ', tf.shape(output).eval())
	return output


def cnn_model(uk0, A, f, gridsize,batch): #uk0 is initialized with pre-smoothing
	weights = {
		#size of convolution based on current size of A?
		#for now, P is a 6x3 convolution, 1 input, 1 output? (or 3?)
		#filter is [filter_width, in_channels, out_channels]
		'P': tf.Variable(tf.random_normal([gridsize/2, gridsize, gridsize])),
		#for now, R is a 3x6 convolution, 1 input, 1 output?
		'R': tf.Variable(tf.random_normal([gridsize/2, gridsize, gridsize]))
		#more??
		#'out': tf.Variable(tf.)
		}
	biases = {
		'b_P': tf.Variable(tf.random_normal([1])),
		'b_R': tf.Variable(tf.random_normal([1]))
	}

	#reshape input to a 3d tensor, [batch_size, input len, input dim]
	A = tf.reshape(A, shape=[batch, gridsize, gridsize])
	#Convolutional layer
	conv1 = tf.nn.conv1d(A, weights['P']+biases['b_P'], stride=1, padding='same')

	conv2 = tf.nn.conv1d(conv1, weights['R']+biases['b_R'], stride=1, padding='same')

	output = tf.matmul(tf.matmul(conv2, A), conv1)

	return output

def interp_model(A,b, gridsize):
	A = tf.reshape()
	with tf.variable_scope('Restriction Operator') as scope:		
		w0 = tf.get_variable("R", [gridsize + 1, num_hidden_units_1])
		b0 = tf.Variable(tf.zeros(num_hidden_units_1))
		h0 = tf.identity(tf.matmul(x, w0)+b0, name="Hidden1")
		print (h0)

	with tf.variable_scope('Prolongation Operator') as scope:
		w1 = tf.get_variable("P", [num_hidden_units_1, 1])
		b1 = tf.Variable(tf.zeros(1))
		output = tf.identity(tf.matmul(h0, w1)+b1, name="Output")
		
	return output

def try_net_training(A, b, u, level):
	#same x and y placeholders
	if level = 0:
		return numpy.linalg.pinv(A) @ b #will need to fix for batches?
	Ac = interp_model(A,b,gridsize)
	Acinv = try_net_training(Ac, level-1)
	D = 1/tf.matrix_diag_part(A)
	D[::2] = 0 #not right way to do this
	output = P @ Acinv(R @ b) + D * b

	loss = tf.losses.mean_squared_error(output, u)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	train = optimizer.minimize(loss)

	


#NEED TO ADD SECOND HIDDEN LAYER!!! 7:17PM
def network_training(training_vals, actual_outputs, num_hidden_units_1, num_hidden_units_2, learning_rate, batch_size, epoch_num, test_vals, test_outputs, process=False):
	x = tf.placeholder(tf.float32, shape=(None, gridsize+1))
	y_ = tf.placeholder(tf.float32, shape=(None, 1))

	output = build_model(x, num_hidden_units_1, num_hidden_units_2)
	weights = output

	loss = tf.losses.mean_squared_error(output, y_)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	train = optimizer.minimize(loss)

	correct = tf.equal(output, y_)
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
	# train
	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)
	for e in range(epoch_num):
	    #print(e)
	    num_entries = len(training_vals)
	    for b in range(int(num_entries/batch_size)):
	    	sess.run(train, {x: np.array(training_vals[b:b+batch_size]), y_: np.array(actual_outputs[b:b+batch_size])})
	    print('epoch # ', e, 'training_performance =', sess.run(accuracy, {x: np.array(training_vals), y_: np.array(actual_outputs)}))
	    
	    if (process == True):
	     	test_num_entries = len(test_vals)
	     	print('testing_performance', sess.run(accuracy, {x: np.array(test_vals), y_: np.array(test_outputs)}))
	print ('weights ', weights)
	print ('Performance with ', num_hidden_units_1, ' of h1 and ', num_hidden_units_2, ' in h2')
	print('training_performance after all epoch of trial = ', sess.run(accuracy, {x: np.array(training_vals), y_: np.array(actual_outputs)}))
	print('testing_performance after all epoch of trial = ', sess.run(accuracy, {x: np.array(test_vals), y_: np.array(test_outputs)}))
	print ('output = ', sess.run(output, {x: np.array(training_vals), y_: np.array(actual_outputs)}))
	print ('actual values = ', actual_outputs)

network_training(train_input_sets, train_solution_sets, 5, 0, 0.001, 1, 25, test_input_sets, test_solution_sets)
	    
