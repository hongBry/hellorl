# Author: Taoz
# Date  : 8/29/2018
# Time  : 3:08 PM
# FileName: config.py


import time

"""experiment"""
# PRE_TRAIN_MODEL_FILE = None
PRE_TRAIN_MODEL_FILE = '/home/hongruying/hellorl/model/net_riverraid_dqn_20180913_002024_20180913_105702.model'
# PRE_TRAIN_MODEL_FILE = 'D:\\software_data\\seekloud\\hellorl\\model\\net_riverraid_dqn_20180913_002024_20180913_105702.model'
EPOCH_NUM = 80
EPOCH_LENGTH = 30000
RANDOM_SEED = int(time.time() * 1000) % 100000000

"""game env"""
GAME_NAME = 'riverraid'
# GAME_NAME = 'breakout'
ACTION_NUM = 18
OBSERVATION_TYPE = 'image'  # image or ram
CHANNEL = 3
WIDTH = 160
HEIGHT = 210
FRAME_SKIP = 4

"""player"""
TRAIN_PER_STEP = 4

"""replay buffer"""
PHI_LENGTH = 4
BUFFER_MAX = 100000
# BUFFER_MAX = 200000
BEGIN_RANDOM_STEP = 1000
if PRE_TRAIN_MODEL_FILE is not None:
    BEGIN_RANDOM_STEP = 100

"""q-learning"""
DISCOUNT = 0.90
EPSILON_MIN = 0.15
EPSILON_START = 1.0
EPSILON_DECAY = 100000
if PRE_TRAIN_MODEL_FILE is not None:
    EPSILON_MIN = 0.15
    EPSILON_START = 0.2
    EPSILON_DECAY = 200

UPDATE_TARGET_BY_EPISODE_END = 50
UPDATE_TARGET_BY_EPISODE_BEGIN = 5
UPDATE_TARGET_DECAY = 100  # update UPDATE_TARGET_DECAY times to get to UPDATE_TARGET_BY_EPISODE_END
UPDATE_TARGET_RATE = (UPDATE_TARGET_BY_EPISODE_END - UPDATE_TARGET_BY_EPISODE_BEGIN) / UPDATE_TARGET_DECAY + 0.000001

LEARNING_RATE = 0.0025
WEIGHT_DECAY = 0.0
# GRAD_CLIPPING_THETA = 0.01
GRAD_CLIPPING_THETA = 0.01

POSITIVE_REWARD = 1
NEGATIVE_REWARD = -1

"""OTHER"""
# MODEL_PATH = '/home/zhangtao/model_file/hello_rl'
MODEL_PATH = '/home/hongruying/hellorl/model'
MODEL_FILE_MARK = 'riverraid_dqn_'
BEGIN_TIME = time.strftime("%Y%m%d_%H%M%S")

print('\n\n\n\n++++++++++++++++ edited time: 2018-09-05 18:17 ++++++++++++++++++')
print('GAME_NAME:', GAME_NAME)
