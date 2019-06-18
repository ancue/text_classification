import itertools
import tensorflow as tf
import numpy as np
import os
from text_cnn import TextCNN

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

def main(argv=None):
    data_path = "./id_data"

    # step 1. Loading Data
    max_sequence_length = None
    data_dict = dict()
    for data_type in ["train", "dev", "test"]:
        label_list = list()
        sequence_list = list()
        with open(os.path.join(data_path, data_type), 'r') as ftrain:
            for line in ftrain:
                fields = line.strip().split()
                label = [0, 0]
                label[int(fields[0])] = 1 # label [1,0] 或 [0,1]
                label_list.append(label) # 标签
                sequence_list.append(fields[1:]) # 句子

        padded_sequences = tf.contrib.keras.preprocessing.sequence.pad_sequences(
            sequence_list,
            padding="post",
            value=0,
            maxlen=max_sequence_length,
        )
        if max_sequence_length is None:
            max_sequence_length = padded_sequences.shape[1]
        data_dict[data_type] = [np.asarray(label_list), padded_sequences]

    # step 2. Build model
    model_config = {
        "vocab_size": max(itertools.chain(*data_dict["train"][1])) + 1,# 词的最大编号加1，即为字典的大小
        "max_sequence_length": data_dict["train"][1].shape[1],
        "emb_dim": 100,
        "channels": 128,# 每种尺寸卷积核的个数
        "filter_sizes": [3, 4, 5], # text_cnn 使用不同的卷积核大小
        "keep_prob": 0.8,
        "batch_size": 64,
        "learning_rate": 3e-4,
    }
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session = tf.Session(config=session_conf)
        with session.as_default():
            cnn = TextCNN(model_config)
            cnn.initialize(session)
            cnn.train(
                session, 
                data_dict["train"][1],
                data_dict["train"][0],
                data_dict["dev"][1],
                data_dict["dev"][0],
            )
            print(cnn.evaluate(session, data_dict["test"][1], data_dict["test"][0]))

if __name__ == '__main__':
    tf.app.run()

"""
原始代码
验证集  97.39%
测试集  95.63%

学习率衰减
验证集  96.62%
测试集  94.32%


"""