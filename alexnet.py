# 输入数据
from tensorflow.examples.tutorials.mnist import input_data

import argparse
import sys

import tensorflow as tf

FLAGS = None

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
sess = tf.InteractiveSession()

# 定义网络参数
learning_rate = 0.0005
training_iters = 100000
batch_size = 64
display_step = 20

# 定义网络参数
n_input = 784  # 输入的维度
n_classes = 10  # 标签的维度
dropout_conv = 1.  # Dropout 的概率
dropout_ac = 0.5

# 定义一些标量的summary
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def alex_net_mnist():

    def dropout(layer_cnt, i):
        return 0.5 * i / (layer_cnt + 1)

    # 输入
    x = tf.placeholder(tf.float32, [None, n_input])

    # 输出
    y = tf.placeholder(tf.float32, [None, n_classes])

    #dropout
    keep_prob = tf.placeholder(tf.float32)


    # 卷积操作                                         输入图像，卷积核，卷积步长，填补类型（same和valid）
    def conv2d(name, l_input, w, b):
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'), b), name=name)

    # 最大下采样操作
    def max_pool(name, l_input, k):
        return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)


    # 归一化操作
    def norm(name, l_input, lsize=4):
        return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)


    # 定义整个网络
    def alex_net(_X, _weights, _biases, _dropout):
        # 向量转为矩阵
        _X = tf.reshape(_X, shape=[-1, 28, 28, 1])
        tf.summary.image('input', _X, 10)

        with tf.name_scope('conv_layer1'):
            with tf.name_scope('conv'):
                # 卷积层1
                conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])
            with tf.name_scope('max_pooling'):
                # 下采样层
                pool1 = max_pool('pool1', conv1, k=2)
                # 归一化层
            with tf.name_scope('normaliztion'):
                norm1 = norm('norm1', pool1, lsize=4)
                # Dropout
            with tf.name_scope('dropout'):
                norm1 = tf.nn.dropout(norm1, _dropout)
                tf.summary.histogram('conv1', norm1)

        # 卷积层2
        with tf.name_scope('conv_layer1'):
            with tf.name_scope('conv'):
                conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
            with tf.name_scope('max_pooling'):
                # 下采样
                pool2 = max_pool('pool2', conv2, k=2)
            with tf.name_scope('normaliztion'):
                # 归一化
                norm2 = norm('norm2', pool2, lsize=4)
            with tf.name_scope('dropout'):
                # Dropout
                norm2 = tf.nn.dropout(norm2, _dropout)

        # 卷积层3
        conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
        # 下采样
        pool3 = max_pool('pool3', conv3, k=2)
        # 归一化
        norm3 = norm('norm3', pool3, lsize=4)
        # Dropout
        norm3 = tf.nn.dropout(norm3, _dropout)

        # 卷积层3
        conv4 = conv2d('conv4', norm3, _weights['wc4'], _biases['bc4'])
        # 下采样
        pool4 = max_pool('pool4', conv4, k=2)
        # 归一化
        norm4 = norm('norm4', pool4, lsize=4)
        # Dropout
        norm4 = tf.nn.dropout(norm4, _dropout)


        # 全连接层，先把特征图转为向量
        dense1 = tf.reshape(norm4, [-1, _weights['wd1'].get_shape().as_list()[0]])
        dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='ac1')

        # tensorboard记录
        tf.summary.histogram('ac1', dense1)
        # 全连接层
        dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='ac2')  # Relu activation
        tf.summary.image('dense2', _X, 10)
        # tensorboard记录
        tf.summary.histogram('ac2', dense2)

        # 网络输出层
        out = tf.matmul(dense2, _weights['out']) + _biases['out']
        return out

    # 存储所有的网络参数
    with tf.name_scope('weights'):
        weights = {
            'wc1': tf.Variable(tf.random_normal([3, 3, 1, 32])),        # 按正态分布初始化权重  卷积层权重指定了filter的大小以及输入输出的数量
            'wc2': tf.Variable(tf.random_normal([3, 3, 32, 64])),
            'wc3': tf.Variable(tf.random_normal([3, 3, 64, 128])),
            'wc4': tf.Variable(tf.random_normal([3, 3, 128, 256])),
            'wd1': tf.Variable(tf.random_normal([2*2*256, 1024])),      # 全连接层将特征图转化为向量
            'wd2': tf.Variable(tf.random_normal([1024, 1024])),
            'out': tf.Variable(tf.random_normal([1024, 10]))
        }
        variable_summaries(weights['wc1'])
        variable_summaries(weights['wc2'])
        variable_summaries(weights['wc3'])
        variable_summaries(weights['wc4'])
        variable_summaries(weights['wd1'])
        variable_summaries(weights['wd2'])
        variable_summaries(weights['out'])

    with tf.name_scope('biases'):
        biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bc3': tf.Variable(tf.random_normal([128])),
            'bc4': tf.Variable(tf.random_normal([256])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'bd2': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([n_classes]))  #最后分类的类别数量
        }
        variable_summaries(biases['bc1'])
        variable_summaries(biases['bc2'])
        variable_summaries(biases['bc3'])
        variable_summaries(biases['bc4'])
        variable_summaries(biases['bd1'])
        variable_summaries(biases['bd2'])
        variable_summaries(biases['out'])


    # 构建模型
    pred_model = alex_net(x, weights, biases, keep_prob)


    # 定义损失函数和学习步骤
    with tf.name_scope('cross_entropy'):
        # tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.softmax(y)),reduction_indices=[1]))  #正确的计算Xentropy方法
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_model, labels=y))
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    # 测试网络
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_pred = tf.equal(tf.argmax(pred_model,1), tf.argmax(y,1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)


    # 初始化所有的共享变量 v1.0更新

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
    tf.global_variables_initializer().run()

    # 开启一个训练
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # 获取批数据
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout_conv})
        if step % display_step == 0:

            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

             # 计算精度
            summary, acc = sess.run([merged, accuracy], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.}, options=run_options, run_metadata=run_metadata)
            test_writer.add_summary(summary, step)
            # 计算损失值
            summary, loss = sess.run([merged, cross_entropy], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.}, options=run_options, run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
            train_writer.add_summary(summary, step)
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        step += 1
    print("优化完成!")
    # 计算测试精度
    acc = sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.})
    print("测试精度:%s" % acc)

    train_writer.close()
    test_writer.close()


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    alex_net_mnist()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/mnist/logs/mnist_alexnet_with_summaries',
                        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


