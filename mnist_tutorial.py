import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# input data

mnist = input_data.read_data_sets('datasets/MNIST_data/', one_hot=True)

x = tf.placeholder(tf.float32, [None, 784], name='x')

if False:

    # simple logistic regression model

    with tf.name_scope('model') as scope:
        W = tf.Variable(tf.zeros([784, 10]), name='weights')
        b = tf.Variable(tf.zeros([10]), name='biases')
        y = tf.nn.softmax(tf.matmul(x, W) + b, name='y')

    # training

    y_ = tf.placeholder(tf.float32, [None, 10], name='y_')
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        summary_writer = tf.train.SummaryWriter('tensorboards/mnist_experiments/run_1', sess.graph)
        sess.run(init)

        print 'training...'

        for i in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

            if i % 10 == 0:
                print i, 'training steps completed'

        print 'finished training'

        # evaluation

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

else:

    # convolutional neural network model

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)


    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')


    with tf.name_scope('conv_model') as scope:
        # 1st convolutional layer

        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])

        x_image = tf.reshape(x, [-1, 28, 28, 1])

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        # 2nd convolutional layer

        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        # densely connected layer

        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # dropout

        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # readout layer

        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # train

    y_ = tf.placeholder(tf.float32, [None, 10], name='y_')

    sess = tf.InteractiveSession()

    summary_writer = tf.train.SummaryWriter('tensorboards/mnist_experiments/run_2', sess.graph)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.initialize_all_variables())

    for i in range(1000):   # was 20000
        batch = mnist.train.next_batch(50)

        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})

            print("step %d, training accuracy %g" % (i, train_accuracy))

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    # evaluate

    for i in range(5):
        batch = mnist.test.next_batch(50)

        print("%d: test accuracy %g" % (i, accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})))
