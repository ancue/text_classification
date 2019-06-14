import os
import numpy as np
import tensorflow as tf
from eval.evaluate import accuracy
from tensorflow.contrib import slim
from loss.loss import cross_entropy_loss

class FastText():
    def __init__(self,
                 num_classes,
                 seq_length,
                 vocab_size,
                 embedding_dim,
                 learning_rate,
                 learning_decay_rate,
                 learning_decay_steps,
                 epoch,
                 dropout_keep_prob
                 ):
        self.num_classes = num_classes
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.learning_decay_rate = learning_decay_rate
        self.learning_decay_steps = learning_decay_steps
        self.epoch = epoch
        self.dropout_keep_prob = dropout_keep_prob
        self.input_x = tf.placeholder(tf.int32, [None, self.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name='input_y')
        self.model()

    def model(self):
        with tf.name_scope("embedding"):
            self.embedding = tf.get_variable("embedding",[self.vocab_size, self.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)

        with tf.name_scope("dropout"):
            dropout_output = tf.nn.dropout(embedding_inputs, self.dropout_keep_prob)

        with tf.name_scopr("average"):
            mean_sentence = tf.reduce_mean(dropout_output, axis=1)

        # 输出层
        with tf.name_scope("score"):
            self.logits = tf.layers.dense(mean_sentence, self.num_classes, name="dense_layer")

        # 损失函数
        self.loss = cross_entropy_loss(logits=self.logits, labels = self.input_y)

        # 优化器
        self.global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(self.learning_decay_rate,
                                                   self.global_step,
                                                   self.learning_decay_steps,
                                                   staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.optim = slim.learning.create_train_op(total_loss = self.loss,
                                                   optimizer = optimizer,
                                                   update_ops = update_ops)

        # 准确率
        self.acc = accuracy(logits = self.logits, labels = self.input_y)

    def fit(self,train_x, train_y, val_x, val_y, batch_size):
        if not os.path.exists('./saves/fasttext'): os.makedirs('./saves/fasttext')
        if not os.path.exists('./train_logs/fasttext'): os.makedirs('./train_logs/fasttext')

        train_steps = 0
        best_val_acc = 0

        tf.summary.scalar('val_loss',self.loss)
        tf.summary.scalar('val_acc',self.acc)
        merged = tf.summary.merge_all()

