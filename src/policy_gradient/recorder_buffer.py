# Author  : hong
# Date    : 2018/9/15
# Time    : 21:43
# File    : recorder_buffer.py

import numpy as np
from src.policy_gradient.config import *

floatX = 'float32'

class RecorderBuffer(object):
    start_mark = 0
    size = 0

    episode_state = []
    episode_actions = []
    episode_rewards = []
    episode_R = []

    def __init__(self, height, width, channel, phi_length=4):
        self.width = width
        self.height = height
        self.channel = channel
        self.phi_length = phi_length

    def reset(self, start_mark):
        self.episode_state = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_R = []
        self.size = 0
        self.start_mark = start_mark

    def add_sample(self, state, action, reward, terminal):
        self.episode_state.append(state.transpose(2, 0, 1))
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        self.episode_R.append(-1000.0)

        self.size += 1
        if terminal:
            self._discount_and_norm_rewards()


    def _discount_and_norm_rewards(self):
        cumulative = 0
        for t in reversed(range(len(self.episode_rewards))):
            if self.episode_rewards[t] != 0:
                cumulative = 0
            cumulative = cumulative * DISCOUNT + self.episode_rewards[t]
            self.episode_R[t] = cumulative

        self.episode_R -= np.mean(self.episode_R)
        self.episode_R /= np.std(self.episode_R)

    def data_iter(self, batch_size):
        imgs = np.array(self.episode_state)
        for i in range(self.start_mark, self.size, batch_size):
            end = min(i + batch_size, self.size)
            states = np.zeros(shape=(end - i,
                                     self.phi_length,
                                     self.channel,
                                     self.height,
                                     self.width
                                     ))
            actions = np.zeros((end - i, 1), dtype='int32')
            rewards = np.zeros((end - i, 1), dtype=floatX)
            R = np.zeros((end - i, 1), dtype=floatX)

            count = 0
            for index in range(i, end):
                all_indexs = np.arange(index - self.phi_length + 1, index + 1)
                states[count] = imgs.take(all_indexs, axis=0, mode='wrap')
                actions[count] = self.episode_actions[index]
                rewards[count] = self.episode_rewards[index]
                R[count] = self.episode_R[index]

                count += 1

            yield states, actions, rewards, R

    def phi(self, img):
        """Return a phi (sequence of image frames), using the last phi_length -
        1, plus img.

        """
        img = img.transpose(2, 0, 1)
        indexes = np.arange(self.size - self.phi_length + 1, self.size)

        phi = np.empty((self.phi_length, self.channel, self.height, self.width), dtype=floatX)
        count = 0
        for index in indexes:
            phi[count] = self.episode_state[index]
            count += 1

        phi[-1] = img
        return phi


def test_recorder_buffer():
    rng = np.random.RandomState()
    buff = RecorderBuffer(5,3,1)
    buff.reset(10)

    for i in range(100):
        img = np.arange(i, i + 15, dtype=np.uint8).reshape(5, 3, 1)
        buff.add_sample(img, i, i + 0.1, False)

    img = np.arange(100, 100 + 15, dtype=np.uint8).reshape(5, 3, 1)
    buff.add_sample(img, 100, 100 + 0.1, True)

    img = np.arange(101, 101 + 15, dtype=np.uint8).reshape(5, 3, 1)
    print(buff.phi(img))

    print('-------------------')
    print(buff.start_mark)
    print('-------------------')
    print(buff.size)

    for states, actions, rewards, Rs in buff.data_iter(10):
        print('===================')
        print('-------------------')
        print(states.shape)
        print('-------------------')
        print(actions)
        print('-------------------')
        print(rewards)
        print('-------------------')
        print(Rs)
        print('===================')

if __name__ == '__main__':
    test_recorder_buffer()









