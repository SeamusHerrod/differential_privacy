#!/usr/bin/env python3
import os
import csv
import math
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import numpy as np

DATA_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'iris.data')

# Load iris; file format: sepal length, sepal width, petal length, petal width, class

def load_iris(path):
    X = []
    y = []
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            if len(row) < 5: continue
            try:
                feats = [float(row[i]) for i in range(4)]
                lab = row[4].strip()
                X.append(feats)
                y.append(lab)
            except:
                continue
    return np.array(X), np.array(y)

# Build indexes: 1-10,51-60,101-110 (1-based) -> convert to 0-based indices
TEST_RANGES = [(1,10),(51,60),(101,110)]

def make_test_indices():
    inds = []
    for a,b in TEST_RANGES:
        for i in range(a-1,b):
            inds.append(i)
    return inds

if __name__ == '__main__':
    X,y = load_iris(DATA_FILE)
    n = len(y)
    test_inds = make_test_indices()
    train_inds = [i for i in range(n) if i not in test_inds]

    X_train = X[train_inds]
    y_train = y[train_inds]
    X_test = X[test_inds]
    y_test = y[test_inds]

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    clf = GaussianNB()
    clf.fit(X_train, y_train_enc)

    y_pred = clf.predict(X_test)

    acc = (y_pred == y_test_enc).mean()

    print('Training size:', len(train_inds))
    print('Test size:', len(test_inds))
    print('Accuracy on test set: {:.4f}'.format(acc))
    print()
    print('Index	True	Predicted')
    for idx, true_l, pred_l in zip(test_inds, y_test, le.inverse_transform(y_pred)):
        print(f'{idx+1}\t{true_l}\t{pred_l}')
