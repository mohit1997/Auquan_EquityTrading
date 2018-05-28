import tensorflow as tf
import numpy as np
import pandas as pd
from time import ctime
import os
import glob

learning_rate = 1e-4
batch_size = 64
mean_win = 50
shift_win = 10
window = 1050
inp_window = 1000
stride = 10
save_path = "model/"

np.random.seed(0)
tf.set_random_seed(0)

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
    	# if(start_idx + batchsize >= inputs.shape[0]):
    	# 	break;

        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(
        a, shape=(nrows, L), strides=(S * n, n), writeable=False)


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def fc(x, h_in, h_out, relu=False):
	W = weight_variable([h_in, h_out])
	b = bias_variable([h_out])

	y = tf.matmul(x, W) + b
	if relu:
		return tf.nn.relu(y)
	return y

def max_pool_2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 1],
                        strides=[1, 2, 1], padding='SAME')


def nn(x):# x - input_placeholder, y - ouput_placeholder
	# inp = tf.expand_dims(x, 2)
	inp = x

	h_conv1 = tf.layers.conv1d(inp, filters=10, kernel_size=(10), padding='SAME', dilation_rate=(5), activation=tf.nn.relu)
	b_1 = tf.layers.batch_normalization(h_conv1, axis=-1)
	p_1 = tf.layers.max_pooling1d(b_1, pool_size=[2], strides=[2])
	h_conv1 = tf.layers.conv1d(p_1, filters=20, kernel_size=(10), padding='SAME', dilation_rate=(5), activation=tf.nn.relu)
	b_1 = tf.layers.batch_normalization(h_conv1, axis=-1)
	p_1 = tf.layers.max_pooling1d(b_1, pool_size=[2], strides=[2])

	h_conv2 = tf.layers.conv1d(p_1, filters=50, kernel_size=(5), padding='SAME', dilation_rate=(2), activation=tf.nn.relu)
	b_2 = tf.layers.batch_normalization(h_conv2, axis=-1)
	p_2 = tf.layers.max_pooling1d(b_2, pool_size=[2], strides=[2])
	h_conv2 = tf.layers.conv1d(p_2, filters=50, kernel_size=(5), padding='SAME', dilation_rate=(2), activation=tf.nn.relu)
	b_2 = tf.layers.batch_normalization(h_conv2, axis=-1)
	p_2 = tf.layers.max_pooling1d(b_2, pool_size=[2], strides=[2])

	#p_2 = p_1

	h_conv3 = tf.layers.conv1d(p_2, filters=1, kernel_size=(1), padding='SAME', activation=tf.nn.relu)
	b_3 = tf.layers.batch_normalization(h_conv3, axis=-1)
	p_3 = tf.layers.max_pooling1d(b_3, pool_size=[2], strides=[2])

	
	flatten = tf.contrib.layers.flatten(p_3)

	shp = flatten.get_shape().as_list()
	h1 = fc(flatten, shp[1], 10, relu=True)
	b_fc1 = tf.layers.batch_normalization(h1, axis=-1)
	
	h2 = fc(b_fc1, 10, 5)
	b_fc2 = tf.layers.batch_normalization(h2, axis=-1)
	h3 = fc(b_fc2, 5, 1)
	return h3

def accuracy(y, labels):
	lab = np.int32(labels)
	# print(y)
	y = 1*(y > 0.5)

	c = np.sum(y == lab)

	print(c/np.float(len(y)))


def mean_var_norm(x): #mean variance norm x: np array
	return  (x - np.mean(x, axis=0))/(2*np.std(x, axis=0))


def split(X, Y):
	size = len(X)
	# shuffle dataset
	indices = np.arange(size)
	#np.random.shuffle(indices)
	# print(indices.shape)
	X = X[indices]
	Y = Y[indices]
	# print("###############################")
	# print(y_train.shape, x_train.shape)

	split = int(np.floor(0.7*len(X)))

	x_train = X[:split]
	y_train = Y[:split]

	indices = np.arange(len(x_train))
	np.random.shuffle(indices)
	x_train = x_train[indices]
	y_train = y_train[indices]

	x_val = X[split:]
	y_val = Y[split:]

	return x_train, y_train, x_val, y_val

def loader(path):
	dat = pd.read_csv(path)
	dat = dat.dropna(axis=0, how='any')

	data_series = dat['stockVWAP']

	data_series = data_series[::4]

	X = np.array(pd.Series(data_series[mean_win:]))
	# X = mean_var_norm(X)
	mean = pd.rolling_mean(pd.Series(data_series), window=mean_win)
	mean = np.array(mean[mean_win:])

	size = np.int32(mean_win/2)
	mean1 = np.array(pd.rolling_mean(pd.Series(data_series), window=size))[mean_win:]

	size = np.int32(mean_win/4)
	mean2 = np.array(pd.rolling_mean(pd.Series(data_series), window=size))[mean_win:]

	#mean = mean_var_norm(mean)
	diff = np.array(pd.Series(data_series)) - np.roll(np.array(pd.Series(data_series)), shift=shift_win)
	diff = diff[mean_win:]
	#diff = mean_var_norm(diff)
	# print(mean.shape)
	# print(X.shape)

    # print(X.shape)
	# # downsample
	# X = X[::20]
	# mean = mean[::20]
	# diff = diff[::20]


	X = strided_app(X, window + 1, stride)
	X1 = strided_app(mean, window + 1, stride)
	X2 = strided_app(diff, window + 1, stride)

	f1 = strided_app(mean1, window + 1, stride)
	f2 = strided_app(mean2, window + 1, stride)

	x_train1 = X1[:, :inp_window]
	x_train2 = X[:, :inp_window]
	x_train3 = X2[:, :inp_window]
	f1 = f1[:, :inp_window]
	f2 = f2[:, :inp_window]
	# print(np.stack((x_train1, x_train2, x_train3), axis=2).shape)

	x_train = np.stack((x_train1, x_train2, x_train3, f1, f2), axis=2)
	x_train = mean_var_norm(x_train)
	# print(x_train.shape)
	# y_train = 1*(X[:, -1:] - X[:, -2:-1] > 0.0)
	y_train = (X1[:, inp_window:inp_window+1] - X1[:, -2:-1])
	# y_train = mean_var_norm(y_train)
	# print(y_train.shape)


	x_train, y_train, x_val, y_val = split(x_train, y_train)
	return x_train, y_train, x_val, y_val

def main():

	#x_train1, y_train1, x_val1, y_val1 = loader("parsedData/ADANIPORTS.csv")
	#x_train2, y_train2, x_val2, y_val2 = loader("parsedData/ADANIPORTS.csv")
	list_stocks = glob.glob('parsedData/*.csv')

	files = ["ADANIPORTS.csv", "ADANIENT.csv", "ADANIPOWER.csv"]
	files = list_stocks[:]

	xtl = []
	ytl = []
	xsl = []
	ysl = []

	for i in files:
		# path = os.path.join("parsedData", i)
		path = i
		x_train1, y_train1, x_val1, y_val1 = loader(path)
		print(x_train1.shape)
		xtl.append(x_train1)
		ytl.append(y_train1)
		xsl.append(x_val1)
		ysl.append(y_val1)

	x_train = np.concatenate(xtl, axis=0)
	y_train = np.concatenate(ytl, axis=0)
	x_val = np.concatenate(xsl, axis=0)
	y_val = np.concatenate(ysl, axis=0)

	print(y_train.shape)

	x = tf.placeholder(tf.float32, [None, inp_window, 5])
	y_ = tf.placeholder(tf.float32, [None, 1])

	y = nn(x)

	# mse = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y))
	mse1 = tf.reduce_mean(tf.losses.mean_squared_error(y_, y))
	mse2 = tf.reduce_mean(tf.losses.mean_squared_error(tf.sign(y_), tf.sign(y)))
	accuracy = (tf.equal(tf.sign(y), tf.sign(y_)))
	accuracy = tf.cast(accuracy, tf.float32)
	accuracy = tf.reduce_mean(accuracy)
	loss = mse1
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	# accuracy =

	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)
	saver = tf.train.Saver()

	sess.run(tf.global_variables_initializer())
	for i in range(100):
		# print(X.shape)
		# print(X.dtype)
		for batch_x, batch_y in iterate_minibatches(x_train, y_train, batchsize=batch_size, shuffle=True):
			sess.run(optimizer, feed_dict={x: batch_x, y_: batch_y})
		if(i%20 == 0):
			print("Epoch", i)
			print("Training Accuracy, Error1, Error 2", sess.run([accuracy, mse1, mse2], feed_dict={x: batch_x, y_: batch_y}))
			# print(x_val)
			# out = sess.run(y, feed_dict={x: x_val})
			ind = np.arange(len(x_val))
			np.random.shuffle(ind)
			a = x_val[ind]
			b = y_val[ind]
			a = a[0:10000]
			b = b[0:10000]
			print(a.shape)
			print("Testing Accuracy, Error1, Error 2", sess.run([accuracy, mse1, mse2], feed_dict={x: a, y_: b}))
			# accuracy(out, y_val)
			model_name = save_path + "model" + ctime() + str(i) + ".ckpt"
			# print(model_name)
			path = saver.save(sess, model_name)
 			print("Model saved in path: %s" % path)


if __name__ == "__main__":
	main()
