# load MNIST data
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import sys
import tensorflow as tf

FLAGS = None


# start tensorflow interactiveSession

#整体网络
def conv():
    mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
    sess = tf.InteractiveSession()

    #summery
    def variable_summaries(var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)  #均值

            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))  #标准差

            tf.summary.scalar('mean', mean)
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    # weight initialization
    def weight_variable(shape):
        weights = tf.truncated_normal(shape, stddev=0.1)
        variable_summaries(weights)
        return tf.Variable(weights)

    def bias_variable(shape):
        bias = tf.constant(0.1, shape = shape)
        variable_summaries(bias)
        return tf.Variable(bias)

    # convolution
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=False)
    # pooling

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Create the model
    # placeholder
    x = tf.placeholder("float", [None, 784])
    y_ = tf.placeholder("float", [None, 10])
    # variables
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.nn.softmax(tf.matmul(x,W) + b)

    # first convolutinal layer
    w_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # second convolutional layer
    w_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # densely connected layer
    w_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    # dropout
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # readout layer
    w_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    with tf.name_scope('softmax'):
        softmax = y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
        tf.summary.histogram('softmax', softmax)

    # train and evaluate the model
    with tf.name_scope('cross_entropy'):
        cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    tf.summary.scalar('cross_entropy', cross_entropy)

    # 梯度下降，可只有选择梯度下降的方法
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(cross_entropy)

    # 预测&正确率
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
    tf.global_variables_initializer().run()

    for i in range(FLAGS.max_steps):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob:1.0})
            summary, acc = sess.run([merged, accuracy], feed_dict={x: batch[0], y_: batch[1], keep_prob:1.0})
            test_writer.add_summary(summary, i)
            print("步数%d, 训练精度 %g" %(i, train_accuracy))

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    # print("测试精度 %g" %accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))
    summary, _ = sess.run([merged, accuracy], feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob:1.0})
    test_writer.add_summary(summary, i)


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    conv()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='迭代次数')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='初始学习率')
    parser.add_argument('--dropout', type=float, default=0.9,
                        help='dropout')
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/conv_summery/input_data',
                        help='存input数据的文件路径')
    parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/conv_summery/logs/summery',
                        help='日志文件')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)