# Author  : hong
# Date    : 2018/9/15
# Time    : 19:29
# File    : policy_network.py

import time
from mxnet import gluon, nd, init, autograd
import numpy as np
import src.utils as g_utils
from src.policy_gradient.config import *

class PolicyGradient(object):
    def __init__(self, ctx, input_sample, model_file=None):
        self.ctx = ctx
        self.policy_net = self.get_net(ACTION_NUM, input_sample)

        if model_file is not None:
            print('%s: read trained model from [%s]' % (time.strftime("%Y-%m-%d %H:%M:%S"), model_file))
            self.policy_net.load_params(model_file, ctx=self.ctx)

        self.trainer = gluon.Trainer(self.policy_net.collect_params(), OPTIMIZER,
                                     {'learning_rate': LEARNING_RATE,
                                      'wd': WEIGHT_DECAY})
        self.loss_func = gluon.loss.SoftmaxCrossEntropyLoss()

    def choose_action(self, state):
        shape0 = state.shape
        # print(shape0)
        state = nd.array(state, ctx=self.ctx, dtype=np.float32).reshape((1, -1, shape0[-2], shape0[-1]))
        if PREPRO_STATE:
            last = state[:, 0, :, :]
            cur = state[:, 1, :, :]
            state = cur - last
        else:
            state = state / 255.0
        out = self.policy_net(state)
        up_probability = nd.softmax(out, axis=1)[0][0]
        return up_probability

    def get_net(self, action_num, input_sample):
        net = gluon.nn.Sequential()
        with net.name_scope():
            net.add(
                gluon.nn.Dense(200, activation="relu"),
                gluon.nn.Dense(action_num)
            )
            # net.add(
            #     gluon.nn.Conv2D(channels=32, kernel_size=8, strides=4, activation='relu'),
            #     gluon.nn.Conv2D(channels=64, kernel_size=4, strides=2, activation='relu'),
            #     gluon.nn.Conv2D(channels=64, kernel_size=3, strides=1, activation='relu'),
            #     gluon.nn.Flatten(),
            #     gluon.nn.Dense(512, activation="relu"),
            #     gluon.nn.Dense(action_num, activation="sigmoid")
            # )
        net.initialize(init.Xavier(), ctx=self.ctx)
        net(input_sample)
        return net

    def save_params_to_file(self, model_path, mark):
        time_mark = time.strftime("%Y%m%d_%H%M%S")
        filename = model_path + '/net_' + str(mark) + '_' + time_mark + '.model'
        self.policy_net.save_parameters(filename)
        print(time.strftime("%Y-%m-%d %H:%M:%S"), ' save model success:', filename)

    def train(self, batch_data):
        loss_sum = 0
        with autograd.record():
            final_nodes = []
            step = 0
            for imgs, actions, rewards, Rs in batch_data:
                batch_size = actions.shape[0]
                s = imgs.shape
                states = imgs.reshape((s[0], -1, s[-2], s[-1]))  # batch x (f x C) x H x W
                if PREPRO_STATE:
                    last = states[:, 0, :, :]
                    cur = states[:, 1, :, :]
                    states = cur - last
                    st = nd.array(states, ctx=self.ctx, dtype=np.float32)
                else:
                    st = nd.array(states, ctx=self.ctx, dtype=np.float32) / 255.0

                at = nd.array(actions[:, 0], ctx=self.ctx)
                Rt = nd.array(Rs[:, 0], ctx=self.ctx)
                labels = at.reshape(shape=(batch_size,))
                logits = self.policy_net(st)
                loss = self.loss_func(logits, labels)
                loss = Rt * loss

                loss_sum += nd.sum(nd.abs(loss)).asscalar()
                step += batch_size

                final_nodes.append(loss)
            autograd.backward(final_nodes)
        self.trainer.step(step)

        return loss_sum, step