# Author: Taoz
# Date  : 8/29/2018
# Time  : 3:08 PM
# FileName: config.py


import time
import sys
import configparser
import os


def load_conf(conf_file):
    # 获取当前文件路径
    current_path = os.path.abspath(__file__)
    # config.ini文件路径,获取当前目录的父目录的父目录与congig.ini拼接
    default_conf_file = os.path.join(os.path.abspath(os.path.dirname(current_path)),
                                     'dqn_default_conf.ini')
    print('default conf file:', default_conf_file)

    print('customer_conf_file:', conf_file)

    print(os.path.exists(default_conf_file))
    print(os.path.exists(conf_file))

    config = configparser.ConfigParser(allow_no_value=True, interpolation=configparser.ExtendedInterpolation())

    print(type(default_conf_file))
    print(type(conf_file))

    print(default_conf_file == 'D:\workstation\python\hellorl\src\dqn\dqn_default_conf.ini')
    print(conf_file == 'D:\workstation\python\hellorl\configurations\dqn_conf_1001.ini')
    config.read(default_conf_file)
    # config.read('D:\workstation\python\hellorl\src\dqn\dqn_default_conf.ini')
    config.read(conf_file)
    # config.read('D:\workstation\python\hellorl\configurations\dqn_conf_1001.ini')
    print(config.sections())
    return config['DQN']


customer_conf_file = sys.argv[1]

dqn_conf = load_conf(customer_conf_file)

"""experiment"""
GPU_INDEX = dqn_conf.getint('GPU_INDEX')
PRE_TRAIN_MODEL_FILE = dqn_conf.get('PRE_TRAIN_MODEL_FILE')
EPOCH_NUM = dqn_conf.getint('EPOCH_NUM')
EPOCH_LENGTH = dqn_conf.getint('EPOCH_LENGTH')
RANDOM_SEED = int(time.time() * 1000) % 100000000

"""game env"""
# GAME_NAME = 'riverraid'
GAME_NAME = dqn_conf.get('GAME_NAME')
ACTION_NUM = dqn_conf.getint('ACTION_NUM')
OBSERVATION_TYPE = dqn_conf.get('OBSERVATION_TYPE')
CHANNEL = dqn_conf.getint('CHANNEL')
WIDTH = dqn_conf.getint('WIDTH')
HEIGHT = dqn_conf.getint('HEIGHT')
FRAME_SKIP = dqn_conf.getint('FRAME_SKIP')

"""player"""
TRAIN_PER_STEP = dqn_conf.getint('TRAIN_PER_STEP')

"""replay buffer"""
PHI_LENGTH = dqn_conf.getint('PHI_LENGTH')
BUFFER_MAX = dqn_conf.getint('BUFFER_MAX')
BEGIN_RANDOM_STEP = dqn_conf.getint('BEGIN_RANDOM_STEP')

"""q-learning"""
DISCOUNT = dqn_conf.getfloat('DISCOUNT')
EPSILON_MIN = dqn_conf.getfloat('EPSILON_MIN')
EPSILON_START = dqn_conf.getfloat('EPSILON_START')
EPSILON_DECAY = dqn_conf.getint('EPSILON_DECAY')

UPDATE_TARGET_BY_EPISODE_END = dqn_conf.getint('UPDATE_TARGET_BY_EPISODE_END')
UPDATE_TARGET_BY_EPISODE_BEGIN = dqn_conf.getint('UPDATE_TARGET_BY_EPISODE_BEGIN')
UPDATE_TARGET_DECAY = dqn_conf.getint('UPDATE_TARGET_DECAY')
UPDATE_TARGET_RATE = (UPDATE_TARGET_BY_EPISODE_END - UPDATE_TARGET_BY_EPISODE_BEGIN) / UPDATE_TARGET_DECAY + 0.000001


OPTIMIZER = dqn_conf.get('OPTIMIZER')
LEARNING_RATE = dqn_conf.getfloat('LEARNING_RATE')
WEIGHT_DECAY = dqn_conf.getfloat('WEIGHT_DECAY')
GRAD_CLIPPING_THETA = dqn_conf.getfloat('GRAD_CLIPPING_THETA')

POSITIVE_REWARD = dqn_conf.getfloat('POSITIVE_REWARD')
NEGATIVE_REWARD = dqn_conf.getfloat('NEGATIVE_REWARD')

if PRE_TRAIN_MODEL_FILE is not None:
    EPSILON_MIN = 0.15
    EPSILON_START = 0.2
    EPSILON_DECAY = 200

"""OTHER"""
MODEL_PATH = dqn_conf.get('MODEL_PATH')
MODEL_FILE_MARK = dqn_conf.get('MODEL_FILE_MARK')
BEGIN_TIME = time.strftime("%Y%m%d_%H%M%S")

EDITED_TIME = dqn_conf.get("EDITED_TIME")
REPLAY_PRIORITY = dqn_conf.getboolean("REPLAY_PRIORITY")
DOUBLE_DQN = dqn_conf.getboolean('DOUBLE_DQN')
DUELING_DQN = dqn_conf.getboolean('DUELING_DQN')

print('\n\n\n\n++++++++++++++++ config edited time: %s ++++++++++++++++++' % EDITED_TIME)
print('BEGIN_TIME:', BEGIN_TIME)
print('CONF FILE:', customer_conf_file)
print('GAME_NAME:', GAME_NAME)
print('EPSILON_MIN:', EPSILON_MIN)
print('EPSILON_START:', EPSILON_START)
print('EPSILON_DECAY:', EPSILON_DECAY)

print('--------------------------')

print('configuration:')
for k, v in dqn_conf.items():
    print('[%s = %s]' % (k, v))

print('--------------------------')