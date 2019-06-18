"""
Author: Lynn @ 解惑者学院
代码类型：文本分类示例程序15

说明：
    此为 CNN 实现文本分类的示例程序，供学员做课程作业的实验参考和学习使用。
    请在此代码的基础上进行改进，尝试以下实验：
        1. 基于指数衰减的学习率。
        2. L2 正则化损失增加可调超参。
        3. 使用网上公开的预训练好的 Embedding 向量。
        4. static 和 non-static 向量结合使用。
        5. 尝试实现 Early-Stoping 的训练方式。

"""
import copy
import tensorflow as tf


class TextCNN(object):
    """
    Reference:
        Kim, Y. (2014). Convolutional neural networks for sentence classification.
                        CoRR, abs/1408.5882. (https://arxiv.org/pdf/1408.5882.pdf)
    """

    def __init__(self, model_config):
        self._m_config = copy.deepcopy(model_config)
        self._build_graph()

    def initialize(self, session):
        session.run(tf.global_variables_initializer())
        return self

    def train(self, session, train_sequences, train_labels, dev_sequences, dev_labels, epoches=10):
        batch_size = self._m_config["batch_size"]
        for epoch in range(epoches):
            # 数据循环次数
            for i in range(0, train_sequences.shape[0], batch_size):
                # 每次取一个batch
                feed_dict = {
                    self._m_ph_sequences: train_sequences[i: i + batch_size],
                    self._m_ph_labels: train_labels[i: i + batch_size],
                    self._m_ph_keep_prob: self._m_config["keep_prob"],
                }
                session.run([self._m_train_op], feed_dict=feed_dict)
                global_step = session.run(self._m_global_step)
                if global_step % 20 == 0:
                    dev_acc = self.evaluate(session, dev_sequences, dev_labels)
                    print("Step %d, dev accuracy %.2f%%" % (global_step, dev_acc * 100))

    def evaluate(self, session, sequences, labels):
        correct_samples = 0
        batch_size = self._m_config["batch_size"]
        for i in range(0, sequences.shape[0], batch_size):
            batch_sequences = sequences[i: i + batch_size]
            batch_labels = labels[i: i + batch_size]

            feed_dict = {
                self._m_ph_sequences: batch_sequences,
                self._m_ph_labels: batch_labels,
                self._m_ph_keep_prob: 1.0,
            }
            batch_acc = session.run(self._m_accuracy, feed_dict=feed_dict)
            correct_samples += batch_size * batch_acc
        return correct_samples / sequences.shape[0]

    def _build_graph(self):
        self._learning_rate()
        self._build_placeholder()
        self._build_embedding_feature()
        self._build_cnn_layer()
        self._build_output()
        self._build_evaluation()
        self._build_optimizer()

    def _learning_rate(self):
        self._m_learning_decay_steps = 100
        self._m_learning_decay_rate = 0.96

    def _build_placeholder(self):
        self._m_ph_sequences = tf.placeholder(
            tf.int32,
            shape=(None, None),
            name="ph_sequences"
        )
        self._m_ph_labels = tf.placeholder(
            tf.float32,
            shape=(None, 2),
            name="ph_labels"
        )
        self._m_ph_keep_prob = tf.placeholder(tf.float32, name="ph_keep_prob")

    def _build_embedding_feature(self):
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self._m_char_emb = tf.Variable(
                tf.random_uniform(
                    [self._m_config["vocab_size"], self._m_config["emb_dim"]],
                    -1.0, 1.0
                ),
                name="char_emb"
            )
            emb_sequences = tf.nn.embedding_lookup(self._m_char_emb, self._m_ph_sequences)
            self._m_emb_sequences = tf.expand_dims(emb_sequences, -1)

    def _build_cnn_layer(self):
        cnn_features = list()
        for filter_size in self._m_config["filter_sizes"]:
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [
                    filter_size,
                    self._m_config["emb_dim"],
                    1,
                    self._m_config["channels"],
                ]
                filter_weights = tf.Variable(
                    tf.truncated_normal(filter_shape, stddev=0.1),
                    name="filter_weights"
                )
                conv = tf.nn.conv2d(
                    self._m_emb_sequences,
                    filter_weights,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv2d"
                )

                # Non-linear projection
                bias = tf.Variable(
                    tf.constant(0.1, shape=[self._m_config["channels"]]),
                    name="bias"
                )
                conv_proj = tf.nn.relu(tf.nn.bias_add(conv, bias), name="relu")

                # Max-Pooling
                feature = tf.nn.max_pool(
                    conv_proj,
                    ksize=[1, self._m_config["max_sequence_length"] - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool"
                )
                cnn_features.append(feature)

        self._m_cnn_filter_num = len(self._m_config["filter_sizes"]) * self._m_config["channels"]
        self._m_cnn_features = tf.reshape(
            tf.concat(cnn_features, 3),
            [-1, self._m_cnn_filter_num]
        )

    def _build_output(self):
        with tf.name_scope("cnn_features_dropout"):
            self._m_cnn_features_dropout = tf.nn.dropout(
                self._m_cnn_features,
                self._m_ph_keep_prob
            )

        self._m_l2_loss = tf.constant(0.0)
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[self._m_cnn_filter_num, 2],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
            self._m_l2_loss += tf.nn.l2_loss(W)
            self._m_l2_loss += tf.nn.l2_loss(b)
            self._m_logits = tf.nn.xw_plus_b(self._m_cnn_features_dropout, W, b, name="logits")
            self._m_prediction = tf.argmax(self._m_logits, 1, name="prediction")

    def _build_evaluation(self):
        with tf.name_scope("loss"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self._m_logits,
                labels=self._m_ph_labels,
            )
            self._m_loss = tf.reduce_mean(cross_entropy) + self._m_l2_loss

        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(self._m_prediction, tf.argmax(self._m_ph_labels, 1))
            self._m_accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32),
                name="accuracy"
            )

    def _build_optimizer(self):
        with tf.name_scope("optimizer"):
            self._m_global_step = tf.Variable(0, name="global_step", trainable=False)
            learning_rate = tf.train.exponential_decay(self._m_config["learning_rate"],
                                                       self._m_global_step,
                                                       self._m_learning_decay_steps,
                                                       self._m_learning_decay_rate,
                                                       staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate)

            grads_and_vars = optimizer.compute_gradients(self._m_loss)
            self._m_train_op = optimizer.apply_gradients(
                grads_and_vars,
                global_step=self._m_global_step,
            )
