import numpy as np
import jieba
import sklearn
import sklearn.linear_model as linear_model
import collections
import itertools
import operator
import array
from sklearn.model_selection import GridSearchCV, KFold

def fetch_train_test(data_path, test_size=0.2):
    y = list()
    text_list = list()

    for line in open(data_path, "r", encoding="utf-8").readlines():
        label, text = line[:-1].split("\t", 1)
        text_list.append(list(jieba.cut(text)))
        y.append(int(label))

    return sklearn.model_selection.train_test_split(
        text_list, y, test_size=test_size, random_state=1028
    )


def build_dict(text_list, min_freq=5):
    freq_dict = collections.Counter(itertools.chain(*text_list))
    freq_list = sorted(freq_dict.items(),
                       key=operator.itemgetter(1),
                       reverse=True)
    words, _ = zip(*filter(lambda wc: wc[1] >= min_freq, freq_list))
    return dict(zip(words, range(len(words))))


def text2vec(text_list, word2id):
    X = list()
    for text in text_list:
        vec = array.array('l', [0] * len(word2id))
        for word in text:
            if word not in word2id:
                continue
            vec[word2id[word]] = 1
        X.append(vec)
    return X


def evaluate(model, X, y):
    accuracy = model.score(X, y)
    fpr, tpr, thesholds = sklearn.metrics.roc_curve(y, model.predict_proba(X)[:, 1], pos_label=1)
    return accuracy, sklearn.metrics.auc(fpr, tpr)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = fetch_train_test("data/train.txt")
    word2id = build_dict(X_train, min_freq=3)

    X_train = text2vec(X_train, word2id)
    X_test = text2vec(X_test, word2id)

    params = {
        "penalty": ["l1", "l2"],
        "C": np.arange(1.0, 2.0, 0.1),
    }

    estimator = linear_model.LogisticRegression()

    gsearch = GridSearchCV(estimator, param_grid=params, scoring="roc_auc", cv=3)
    gsearch.fit(X_train, y_train)
    best_model = gsearch.best_estimator_

    print("最好参数", gsearch.best_params_)
    accuracy, auc = evaluate(best_model, X_train, y_train)

    print("训练集正确率：%.4f%%\n" % (accuracy * 100))
    print("训练集AUC值：%.6f\n" % (auc))

    accuracy, auc = evaluate(best_model, X_test, y_test)
    print("验证集正确率：%.4f%%\n" % (accuracy * 100))
    print("验证AUC值：%.6f\n" % (auc))
