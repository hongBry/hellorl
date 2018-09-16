from mxnet.gluon import nn
from mxnet import nd, init

class DDQN_Network(nn.Block):
    def __init__(self, action_num, **kwargs):
        super(DDQN_Network, self).__init__(**kwargs)
        with self.name_scope():
            self.conv_layer = nn.Sequential()
            self.conv_layer.add(
                nn.Conv2D(channels=32, kernel_size=8, strides=4, activation='relu'),
                nn.Conv2D(channels=64, kernel_size=4, strides=2, activation='relu'),
                nn.Conv2D(channels=64, kernel_size=3, strides=1, activation='relu'),
                nn.Flatten(),
                nn.Dense(512, activation="relu")
            )
            self.value_layer = nn.Dense(1)
            self.advantage_layer = nn.Dense(action_num)



    def forward(self, x):
        input_future = self.conv_layer(x)
        v = self.value_layer(input_future)
        a = self.advantage_layer(input_future)
        q = v + (a - nd.mean(a, axis=1, keepdims=True))
        return q

def test_net():
    net = DDQN_Network(18)
    net.initialize(init.Xavier())
    INPUT_SAMPLE = nd.ones(shape=(2,12,210,160))
    print(net(INPUT_SAMPLE))
    net.save_parameters("test.model")
    net1 = DDQN_Network(18)
    net1.load_parameters("test.model")
    print(net1(INPUT_SAMPLE))

if __name__ == '__main__':
    test_net()