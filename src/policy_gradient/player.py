# Author  : hong
# Date    : 2018/9/15
# Time    : 21:08
# File    : player.py


import numpy as np
from src.policy_gradient.config import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
            if True:
                st = self.game.prepro(st)
                st = np.array(st).reshape((PREPRO_HEIGHT, PREPRO_WIDTH, PREPRO_CHANNEL))

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
                time.sleep(0.05)

        if not testing:
            avg_loss, train_count = self._learn(recorder_buffer)
        return episode_step - no_op_steps, episode_reword, episode_score, avg_loss, train_count

    def _choose_action(self, img, recorder_buffer, testing):
        if PREPRO_STATE:
            img = self.game.prepro(img).reshape((PREPRO_HEIGHT, PREPRO_WIDTH, PREPRO_CHANNEL))

        # if testing:
        #     imgs = img.reshape((PREPRO_HEIGHT,PREPRO_WIDTH))
        #     plt.imshow(imgs, cmap=cm.gray)
        #     plt.show()
        phi = recorder_buffer.phi(img)
        up_probability = self.policy_gradient.choose_action(phi)[0]
        if np.random.uniform() < up_probability:
            action = 0
        else:
            action = 1

        return action

    def _learn(self, recorder_buffer):
        # recorder_buffer.size - recorder_buffer.start_mark is all data of one eposide
        batch_data = recorder_buffer.data_iter(recorder_buffer.size - recorder_buffer.start_mark)
        loss_sum, loss_count = self.policy_gradient.train(batch_data)
        return loss_sum / (loss_count + 0.0000001), loss_count
