import os
import gzip
import numpy as np

from urllib import request, error


def vectorized(x):
    out = np.zeros((10, 1))
    out[x] = 1
    return out


def load_data(filename='output'):
    if not os.path.exists('../data_fetch/' + '../data_fetch/' + filename + "ImagesTrain"):
        try:
            request.urlretrieve('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', '../data_fetch/' + filename + "ImagesTrain")
        except error.HTTPError as e:
            print(e.code, e.reason)
            pass

    if not os.path.exists('../data_fetch/' + filename + "LabelsTrain"):
        try:
            request.urlretrieve('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', '../data_fetch/' + filename + "LabelsTrain")
        except error.HTTPError as e:
            print(e.code, e.reason)
            pass

    if not os.path.exists('../data_fetch/' + filename + "ImagesTest"):
        try:
            request.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', '../data_fetch/' + filename + "ImagesTest")
        except error.HTTPError as e:
            print(e.code, e.reason)
            pass

    if not os.path.exists('../data_fetch/' + filename + "LabelsTest"):
        try:
            request.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', '../data_fetch/' + filename + "LabelsTest")
        except error.HTTPError as e:
            print(e.code, e.reason)
            pass
    with gzip.open('../data_fetch/' + filename + "ImagesTrain", 'rb') as file:
        data = np.frombuffer(file.read(), np.uint8, offset=16)
        x_Train = data.reshape(-1, 784, 1) / np.float32(256)

    with gzip.open('../data_fetch/' + filename + "LabelsTrain", 'rb') as file:
        data = np.frombuffer(file.read(), np.uint8, offset=8)
        y_Train = [vectorized(x) for x in data]

    with gzip.open('../data_fetch/' + filename + "ImagesTest", 'rb') as file:
        data = np.frombuffer(file.read(), np.uint8, offset=16)
        x_Test = data.reshape(-1, 784, 1) / np.float32(256)
    with gzip.open('../data_fetch/' + filename + "LabelsTest", 'rb') as file:
        data = np.frombuffer(file.read(), np.uint8, offset=8)
        y_Test = [vectorized(x) for x in data]

    train_data = zip(x_Train, y_Train)
    test_data = zip(x_Test, y_Test)
    return list(train_data), list(test_data)
