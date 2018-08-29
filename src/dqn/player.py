# Author: Taoz
# Date  : 8/25/2018
# Time  : 12:22 PM
# FileName: player.py

import numpy as np
from src.dqn.config import *


class Player(object):
    def __init__(self, game, q_learning, rng):
        self.game = game
        self.action_num = self.game.action_num()  # [0,1,2,..,action_num-1]
        self.q_learning = q_learning
        self.rng = rng
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_rate = (EPSILON_START - EPSILON_MIN) * 1.0 / EPSILON_DECAY

    def run_episode(self, epoch, max_steps, replay_buffer, render=False, random_action=False, testing=False):
        episode_step = 0
        episode_reword = 0
        train_count = 0
        loss_sum = 0
        st = self.game.reset()
        while True:
            # print('run_episode step: %d' % (episode_step))

            if not testing and random_action:
                action = self.game.random_action()
            else:
                action = self._choose_action(st, replay_buffer)

            next_st, reward, episode_done, lives = self.game.step(action)
            terminal = episode_done or episode_step >= max_steps

            replay_buffer.add_sample(st, action, reward, terminal)
            episode_step += 1
            episode_reword += reward
            st = next_st
            if terminal:
                break

            if render:
                self.game.render()
            if not testing and episode_step % TRAIN_PER_STEP == 0 and not random_action:
                # print('-- train_policy_net episode_step=%d' % episode_step)
                imgs, actions, rs, terminal = replay_buffer.random_batch(32)
                # print('img:', imgs.shape, imgs.dtype)
                # print('actions:', actions.shape, actions.dtype)
                # print('rs:', rs.shape, rs.dtype)
                # print('terminal:', terminal.shape, terminal.dtype)
                loss = self.q_learning.train_policy_net(imgs, actions, rs, terminal)
                loss_sum += loss
                train_count += 1
        return episode_step, episode_reword, loss_sum * 1.0 / (train_count + 0.00001)

    def _choose_action(self, img, replay_buffer):
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_rate)
        if self.rng.rand() < self.epsilon:
            action = self.rng.randint(0, self.action_num)
        else:
            phi = replay_buffer.phi(img)
            action = self.q_learning.choose_action(phi)
        return action


if __name__ == '__main__':
    pass
