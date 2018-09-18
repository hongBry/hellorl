# Author  : hong
# Date    : 2018/9/15
# Time    : 19:03
# File    : experiment.py

import time
import gym
import numpy as np
from src.policy_gradient.game_env import GameEnv
import mxnet as mx
import mxnet as nd
from src import utils
import src.ztutils as ztutils
from src.policy_gradient.config import *
from src.policy_gradient.player import Player
from src.policy_gradient.recorder_buffer import RecorderBuffer
from src.policy_gradient.policy_network import PolicyGradient


class Experiment(object):
    ctx = utils.try_gpu(GPU_INDEX)
    if PREPRO_STATE:
        INPUT_SAMPLE = nd.random.uniform(0, 255, (1, 1, PREPRO_HEIGHT, PREPRO_WIDTH), ctx=ctx) / 255.0
    else:
        INPUT_SAMPLE = nd.random.uniform(0, 255, (1, PHI_LENGTH * CHANNEL, HEIGHT, WIDTH), ctx=ctx) / 255.0
    mx.random.seed(RANDOM_SEED)
    rng = np.random.RandomState(RANDOM_SEED)

    def __init__(self, testing=False):
        ztutils.mkdir_if_not_exist(MODEL_PATH)
        self.step_count = 0
        self.episode_count = 0
        self.policy_network = PolicyGradient(ctx=self.ctx,
                                             input_sample=self.INPUT_SAMPLE,
                                             model_file=PRE_TRAIN_MODEL_FILE)
        self.game = GameEnv(game=GAME_NAME,
                            obs_type=OBSERVATION_TYPE,
                            frame_skip=FRAME_SKIP)

        self.player = Player(self.game,
                             self.policy_network,
                             Experiment.rng)
        if PREPRO_STATE:
            self.recorder_buffer = RecorderBuffer(PREPRO_HEIGHT, PREPRO_WIDTH, PREPRO_CHANNEL, PHI_LENGTH)
        else:
            self.recorder_buffer = RecorderBuffer(HEIGHT, WIDTH, CHANNEL, PHI_LENGTH)
        self.testing = testing

    def start_train(self):
        for i in range(1, EPOCH_NUM + 1):
            self._run_epoch(i)
        print('train done.')
        self.game.close()

    def start_test(self):
        assert PRE_TRAIN_MODEL_FILE is not None
        for i in range(1, EPOCH_NUM + 1):
            self._run_epoch(i, render=True)
        print('test done.')
        self.game.close()

    def _run_epoch(self, epoch, render=False):
        step_in_epoch = 0
        reward_in_epoch = 0.0
        score_in_epoch = 0.0
        train_steps_in_epoch = 0

        for eposide in range(1, EPOSIDES_IN_EPOCH + 1):
            t0 = time.time()
            ep_steps, ep_reward, ep_score, avg_loss, train_steps = self.player.run_episode(self.recorder_buffer,
                                                                              render=render,
                                                                              testing=self.testing)
            train_steps_in_epoch += train_steps
            self.step_count += ep_steps
            self.episode_count += 1
            score_in_epoch += ep_score
            step_in_epoch += ep_steps
            reward_in_epoch += ep_reward
            t1 = time.time()

            print(
                'episode [%d], episode step=%d, total_step=%d, time=%.2fs, score=%.2f, ep_reward=%.2f, avg_loss=%.4f, train_steps=%d'
                % (eposide, ep_steps, self.step_count, (t1 - t0), ep_score, ep_reward, avg_loss, train_steps))
            print('')

        self._save_net()
        print('\n%s EPOCH finish [%d], episode=%d, step=%d, avg_step=%d, avg_score=%.2f avg_reward=%.2f, train_steps_in_epoch=%d \n\n\n' %
              (time.strftime("%Y-%m-%d %H:%M:%S"),
               epoch,
               self.episode_count,
               self.step_count,
               step_in_epoch // EPOSIDES_IN_EPOCH,
               score_in_epoch / EPOSIDES_IN_EPOCH,
               reward_in_epoch / EPOSIDES_IN_EPOCH,
               train_steps_in_epoch))

    def _save_net(self):
        if not self.testing:
            self.policy_network.save_params_to_file(MODEL_PATH, MODEL_FILE_MARK + BEGIN_TIME)


def train():
    print(' ====================== START TRAIN ========================')
    exper = Experiment()
    exper.start_train()


def test():
    print(' ====================== START test ========================')
    exper = Experiment(testing=True)
    exper.start_test()

if __name__ == '__main__':
    train()