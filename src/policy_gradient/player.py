# Author  : hong
# Date    : 2018/9/15
# Time    : 21:08
# File    : player.py


import numpy as np
from src.policy_gradient.config import *

class Player(object):
    def __init__(self, game, policy_gradient, rng):
        self.game = game
        self.action_num = self.game.action_num()  # [0,1,2,..,action_num-1]
        self.policy_gradient = policy_gradient
        self.rng = rng

    def run_episode(self, recorder_buffer, render=False, testing=False):
        episode_step = 0
        episode_reword = 0
        episode_score = 0.0
        avg_loss = 0.0
        train_count = 0
        st = self.game.reset()


        # do no operation steps.
        max_no_op_steps = 10
        no_op_steps = self.rng.randint(max(4, PHI_LENGTH), max_no_op_steps)
        recorder_buffer.reset(no_op_steps)


        while True:
            if episode_step < no_op_steps:
                action = 0
                fake_action = 0
            else:
                action = self._choose_action(st, recorder_buffer, testing)
                fake_action = 2 if action == 0 else 3

            next_st, reward, episode_done, lives, score = self.game.step(fake_action)
            terminal = episode_done
            recorder_buffer.add_sample(st, action, reward, terminal)
            if episode_step >= no_op_steps:
                episode_reword += reward
                episode_score += score
            episode_step += 1
            st = next_st
            if terminal:
                break

            if render:
                self.game.render()

        if not testing:
            avg_loss, train_count = self._learn(recorder_buffer)
        return episode_step - no_op_steps, episode_reword, episode_score, avg_loss, train_count

    def _choose_action(self, img, recorder_buffer, testing):
        # self.epsilon = 0.2
        #
        #
        # if not testing and self.rng.rand() < self.epsilon:
        #     action = self.rng.randint(0, self.action_num)
        # else:
        phi = recorder_buffer.phi(img)
        action_prop = self.policy_gradient.choose_action(phi)
        action_prop = action_prop.asnumpy()
        # print(action_prop)
        action = self.rng.choice(range(len(action_prop.ravel())), p=action_prop.ravel())

        return action

    def _learn(self, recorder_buffer):
        batch_data = recorder_buffer.data_iter(32)
        loss_sum = 0
        loss_count = 0
        last_state = None
        for state, actions, rewards, Rs in batch_data:
            loss = self.policy_gradient.train(state, actions, rewards, Rs)
            loss_sum += loss
            loss_count += 1
            # if last_state is not None:
                # x = np.equal(last_state, state)
                # x = x.astype(dtype=np.int32)
                # y = np.ones_like(x)
                # print(y.sum() - x.sum())
                # print(actions.sum())
            # last_state = state
        return loss_sum / (loss_count + 0.0000001), loss_count
