import tensorflow as tf

overfeatfast = tf.Graph()


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)


with overfeatfast.as_default():
	l0_input = tf.placeholder(dtype=tf.float32,
							shape=[None, 221, 221, 3])

	conv1_w = weight_variable([7, 7, 3, 96])

	conv1_o = tf.nn.conv2d(input=l0_input,
						filter=conv1_w,
						strides=[1, 2, 2, 1],
						padding="VALID",
						use_cudnn_on_gpu=True,
						name="conv1"
						)
	pool1_o = tf.nn.max_pool(value=conv1_o,
							 ksize=[1, 3, 3, 1],
							 strides=[1, 3, 3, 1],
							 padding="VALID",
							 name="pool1")

	conv2_w = weight_variable([7, 7, 96, 256])

	conv2_o = tf.nn.conv2d(input=pool1_o,
						filter=conv2_w,
						strides=[1, 1, 1, 1],
						padding="VALID",
						use_cudnn_on_gpu=True,
						name="conv2")
	pool2_o = tf.nn.max_pool(value=conv2_o,
							 ksize=[1, 2, 2, 1],
							 strides=[1, 2, 2, 1],
							 padding="VALID",
							 name="pool2")

	conv3_w = weight_variable([3, 3, 256, 512])
	conv3_o = tf.nn.conv2d(input=pool2_o,
						filter=conv3_w,
						strides=[1, 1, 1, 1],
						padding="SAME",
						use_cudnn_on_gpu=True,
						name="conv3")

	conv4_w = weight_variable([3, 3, 512, 512])
	conv4_o = tf.nn.conv2d(input=conv3_o,
						filter=conv4_w,
						strides=[1, 1, 1, 1],
						padding="SAME",
						use_cudnn_on_gpu=True,
						name="conv4")

	conv5_w = weight_variable([3, 3, 512, 1024])
	conv5_o = tf.nn.conv2d(input=conv4_o,
						filter=conv5_w,
						strides=[1, 1, 1, 1],
						padding="SAME",
						use_cudnn_on_gpu=True,
						name="conv5")

	conv6_w = weight_variable([3, 3, 1024, 1024])
	conv6_o = tf.nn.conv2d(input=conv5_o,
						filter=conv6_w,
						strides=[1, 1, 1, 1],
						padding="SAME",
						use_cudnn_on_gpu=True,
						name="conv6")

	pool3_o = tf.nn.max_pool(value=conv6_o,
							ksize=[1, 3, 3, 1],
							strides=[1, 3, 3, 1],
							padding="VALID",
							name="pool3")

	fc1_o = tf.Variable(initial_value=tf.random_normal([tf.size(pool3_o), 4096]),
						validate_shape=False,
						name="fc1")

	fc2_o = tf.Variable(initial_value=tf.random_normal([tf.size(fc1_o), 4096]),
						validate_shape=False,
						name="fc2")

tf.train.write_graph(overfeatfast.as_graph_def(),
					logdir='./ModelFiles',
					name='OverFeatAccurate.pbtxt',
					as_text=True)
