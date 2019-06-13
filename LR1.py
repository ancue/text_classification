import numpy as np
import sys
import jieba
import sklearn
import sklearn.linear_model as linear_model
import collections
import itertools
import operator
import array


def fetch_train_test(data_path, test_size=0.2):
    y = list()
    text_list = list()

    for line in open(data_path, "r", encoding="utf-8").readlines():
        label, text = line[:-1].split("\t", 1)
        text_list.append(list(jieba.cut(text)))
        y.append(int(label))

    return sklearn.model_selection.train_test_split(text_list, y, test_size=test_size, random_state=0)


def build_dict(text_list, min_freq=5):
    freq_dict = collections.Counter(itertools.chain(*text_list))
    freq_list = sorted(freq_dict.items(), key=operator.itemgetter(1), reverse=True)
    words, _ = zip(*filter(lambda wc: wc[1] >= min_freq, freq_list))
    return dict(zip(words, range(len(words))))


# BOW
# 可以修改为tf-idf
def text2vec(text_list, word2id):
    X = list()
    for text in text_list:
        vec = array.array('l', [0] * len(word2id))  # 类型 long ,l
        for word in text:
            if word not in word2id:
                continue
            vec[word2id[word]] = 1
        X.append(vec)
    return X


def evaluate(model, X, y):
    accuracy = model.score(X, y)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, model.predict_proba(X)[:, 1], pos_label=1)
    return accuracy, sklearn.metrics.auc(fpr, tpr)

if __name__ == "__main__":

    # step 1. 将原始数据拆分成训练集和测试集
    X_train, X_test, y_train, y_test = fetch_train_test("data/train.txt")

    # step 2. 创建字典
    word2id = build_dict(X_train, min_freq=3)

    # step 3. 抽取特征
    X_train = text2vec(X_train, word2id)
    X_test = text2vec(X_test, word2id)

    # step 4. 训练模型
    lr = linear_model.LogisticRegression(C=1)
    lr.fit(X_train, y_train)

    # step 5. 模型评估
    accuracy, auc = evaluate(lr, X_train, y_train)
    sys.stdout.write("训练集正确率：%.4f%%\n" % (accuracy * 100))
    sys.stdout.write("训练集AUC值：%.6f\n" % (auc))

    accuracy, auc = evaluate(lr, X_test, y_test)
    sys.stdout.write("测试集正确率：%.4f%%\n" % (accuracy * 100))
    sys.stdout.write("测试AUC值：%.6f\n" % (auc))