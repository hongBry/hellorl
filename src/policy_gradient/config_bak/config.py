# Author  : hong
# Date    : 2018/9/15
# Time    : 18:36
# File    : config.py


import time

GPU_INDEX = 0

# GAME_NAME = 'riverraid'
GAME_NAME = 'pong'


PRE_TRAIN_MODEL_FILE = None
# PRE_TRAIN_MODEL_FILE = '/home/zhangtao/model_file/hello_rl/net_params_test1_20180831_210601_20180901_213712.model'
# PRE_TRAIN_MODEL_FILE = 'D:\data\\rl\\model\\net_params_test1_20180901_214959_20180902_081326.model'
OBSERVATION_TYPE = 'image'  # image or ram
ACTION_NUM = 2
FRAME_SKIP = 4
EPOCH_NUM = 360
EPOSIDES_IN_EPOCH = 300

PHI_LENGTH = 4
CHANNEL = 3
WIDTH = 160
HEIGHT = 210

DISCOUNT = 0.99
RANDOM_SEED = int(time.time() * 1000) % 100000000


LEARNING_RATE = 0.0025
WEIGHT_DECAY = 0.0

GRAD_CLIPPING_THETA = 0.01

# reward
DEATH_REWARD = -1
POSITIVE_REWARD = 1
NEGATIVE_REWARD = -1

MODEL_PATH = '/home/hongruying/hellor_reply_priority/model'
# MODEL_PATH = 'D:\\software_data\\seekloud\\hellorl\\model'
MODEL_FILE_MARK = 'pong_pg_'
BEGIN_TIME = time.strftime("%Y%m%d_%H%M%S")

print('\n\n\n\n++++++++++++++++ edited time: 22:19 ++++++++++++++++++')
