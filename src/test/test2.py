import numpy as np
from mxnet.gluon import nn
from mxnet import init, nd
import mxnet as mx


def np_take():
    array = np.ones(shape=(10, 10, 10), dtype='int')
    print(np.take(array, [1, 2, 0], axis=0).shape)

    array = np.ones(shape=(10, 10, 10), dtype='int')
    array[0][0][0] = 0
    array[9][0][0] = 3
    print(np.take(array, [10,1, 2], axis=0, mode='clip'))

    array = np.arange(0, 10)
    print(array.take(array[0:-1]))

def get_net():
    net = nn.Sequential()
    with net.name_scope():
        net.add(
            nn.Conv2D(channels=32, kernel_size=8, strides=4, activation='relu'),
            nn.Conv2D(channels=64, kernel_size=4, strides=2, activation='relu'),
            nn.Conv2D(channels=64, kernel_size=3, strides=1, activation='relu'),
            nn.Flatten(),
            nn.Dense(512, activation="relu"),
            nn.Dense(18)
        )
    net.initialize(init.Xavier(),ctx=mx.cpu())
    INPUT_SAMPLE = nd.random.uniform(0, 255, (1, 4 * 3, 210, 160), ctx=mx.cpu()) / 255.0
    print(net.collect_params().keys)
    net(INPUT_SAMPLE)
    # net.save_params("test.model")
    net.load_params("test.model")
    print("finish")



if __name__ == '__main__':
    get_net()