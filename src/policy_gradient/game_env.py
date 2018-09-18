# Author: Taoz
# Date  : 8/25/2018
# Time  : 12:23 PM
# FileName: environment.py

import gym
from gym.envs.atari import AtariEnv
from src.policy_gradient.config import *


class GameEnv(object):
    def __init__(self, game, obs_type, frame_skip):
        self.gym_env = AtariEnv(game=game,
                                obs_type=obs_type,
                                frameskip=frame_skip,
                                repeat_action_probability=0.05)
        self.step_count = 0
        self.gym_env.reset()
        self.lives = self.gym_env.ale.lives()

    def step(self, action):
        self.step_count += 1
        observation, reward, done, _ = self.gym_env.step(action)
        score = reward
        new_lives = self.gym_env.ale.lives()
        # reward = reward / 80
        # reward = max(NEGATIVE_REWARD, min(POSITIVE_REWARD, reward))
        # reward += self.step_count * 0.0002

        # if self.lives > new_lives:
        #     reward = DEATH_REWARD
        self.lives = new_lives
        return observation, reward, done, new_lives, score

    def render(self):
        return self.gym_env.render()

    def random_action(self):
        return self.gym_env.action_space.sample()

    def action_num(self):
        return self.gym_env.action_space.n

    def reset(self, skip_begin_frame=5):
        assert skip_begin_frame > 0
        self.gym_env.reset()
        obs = None
        for _ in range(skip_begin_frame):
            obs, _, _, _ = self.gym_env.step(self.gym_env.action_space.sample())
        self.lives = self.gym_env.ale.lives()
        self.step_count = 0
        return obs

    def close(self):
        self.gym_env.close()

    # preprocessing function only for pong
    def prepro(self, img):  # where I is the single frame of the game as the input
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        # the values below have been precomputed through trail and error by OpenAI team members
        img = img[35:195]  # cropping the image frame to an extent where it contains on the paddles and ball and area between them
        img = img[::2, ::2, 0]  # downsample by the factor of 2 and take only the R of the RGB channel.Therefore, now 2D frame
        img[img == 144] = 0  # erase background type 1
        img[img == 109] = 0  # erase background type 2
        img[img != 0] = 1  # everything else(other than paddles and ball) set to 1
        return img.astype('float').ravel()  # flattening to 1D
