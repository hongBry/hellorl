# Author: Taoz
# Date  : 8/29/2018
# Time  : 3:08 PM
# FileName: config.py


import time

GAME_NAME = 'riverraid'
#  PRE_TRAIN_MODEL_FILE = None
# PRE_TRAIN_MODEL_FILE = '/home/zhangtao/model_file/hello_rl/net_params_test1_20180831_210601_20180901_213712.model'
PRE_TRAIN_MODEL_FILE = 'D:\data\\rl\\model\\net_params_test1_20180901_214959_20180902_081326.model'
OBSERVATION_TYPE = 'image'  # image or ram
FRAME_SKIP = 4
EPOCH_NUM = 360
EPOCH_LENGTH = 30000

PHI_LENGTH = 4
CHANNEL = 3
WIDTH = 160
HEIGHT = 210

BEGIN_RANDOM_STEP = 1000

BUFFER_MAX = 50000
# BUFFER_MAX = 200000
DISCOUNT = 0.90
RANDOM_SEED = int(time.time() * 1000) % 100000000
EPSILON_MIN = 0.15
EPSILON_START = 1.0
EPSILON_DECAY = 100000

if PRE_TRAIN_MODEL_FILE is not None:
    BEGIN_RANDOM_STEP = 100
    EPSILON_MIN = 0.15
    EPSILON_START = 0.2
    EPSILON_DECAY = 20000

TRAIN_PER_STEP = 4
# UPDATE_TARGET_BY_STEP = 30000


UPDATE_TARGET_BY_EPISODE_END = 30
UPDATE_TARGET_BY_EPISODE_BEGIN = 29
UPDATE_TARGET_DECAY = 2  # update UPDATE_TARGET_DECAY times to get to UPDATE_TARGET_BY_EPISODE_END
UPDATE_TARGET_RATE = (UPDATE_TARGET_BY_EPISODE_END - UPDATE_TARGET_BY_EPISODE_BEGIN) / UPDATE_TARGET_DECAY + 0.000001

LEARNING_RATE = 0.005
WEIGHT_DECAY = 0.0

GRAD_CLIPPING_THETA = 0.01

POSITIVE_REWARD = 1
NEGATIVE_REWARD = -1

MODEL_PATH = '/home/zhangtao/model_file/hello_rl'

BEGIN_TIME = time.strftime("%Y%m%d_%H%M%S")

print('\n\n\n\n++++++++++++++++ edited time: 21:45 ++++++++++++++++++')