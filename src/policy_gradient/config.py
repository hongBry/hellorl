# Author  : hong
# Date    : 2018/9/15
# Time    : 18:36
# File    : config.py

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
    return config['PG']


customer_conf_file = sys.argv[1]

pg_conf = load_conf(customer_conf_file)

"""experiment"""
GPU_INDEX = pg_conf.getint('GPU_INDEX')
PRE_TRAIN_MODEL_FILE = pg_conf.get('PRE_TRAIN_MODEL_FILE')
EPOCH_NUM = pg_conf.getint('EPOCH_NUM')
EPOSIDES_IN_EPOCH = pg_conf.getint('EPOSIDES_IN_EPOCH')
RANDOM_SEED = int(time.time() * 1000) % 100000000

"""game env"""
# GAME_NAME = 'riverraid'
GAME_NAME = pg_conf.get('GAME_NAME')
ACTION_NUM = pg_conf.getint('ACTION_NUM')
OBSERVATION_TYPE = pg_conf.get('OBSERVATION_TYPE')
CHANNEL = pg_conf.getint('CHANNEL')
WIDTH = pg_conf.getint('WIDTH')
HEIGHT = pg_conf.getint('HEIGHT')
FRAME_SKIP = pg_conf.getint('FRAME_SKIP')
PHI_LENGTH = pg_conf.getint('PHI_LENGTH')


"""pg"""
DISCOUNT = pg_conf.getfloat('DISCOUNT')

OPTIMIZER = pg_conf.get('OPTIMIZER')
LEARNING_RATE = pg_conf.getfloat('LEARNING_RATE')
WEIGHT_DECAY = pg_conf.getfloat('WEIGHT_DECAY')
GRAD_CLIPPING_THETA = pg_conf.getfloat('GRAD_CLIPPING_THETA')
GAMMA1 = pg_conf.getfloat('GAMMA1')



"""OTHER"""
MODEL_PATH = pg_conf.get('MODEL_PATH')
MODEL_FILE_MARK = pg_conf.get('MODEL_FILE_MARK')
BEGIN_TIME = time.strftime("%Y%m%d_%H%M%S")

EDITED_TIME = pg_conf.get("EDITED_TIME")

print('\n\n\n\n++++++++++++++++ config edited time: %s ++++++++++++++++++' % EDITED_TIME)
print('BEGIN_TIME:', BEGIN_TIME)
print('CONF FILE:', customer_conf_file)
print('GAME_NAME:', GAME_NAME)


print('--------------------------')

print('configuration:')
for k, v in pg_conf.items():
    print('[%s = %s]' % (k, v))

print('--------------------------')

#
# import time
#
# GPU_INDEX = 0
#
# # GAME_NAME = 'riverraid'
# GAME_NAME = 'pong'
#
#
# PRE_TRAIN_MODEL_FILE = None
# # PRE_TRAIN_MODEL_FILE = '/home/zhangtao/model_file/hello_rl/net_params_test1_20180831_210601_20180901_213712.model'
# # PRE_TRAIN_MODEL_FILE = 'D:\data\\rl\\model\\net_params_test1_20180901_214959_20180902_081326.model'
# OBSERVATION_TYPE = 'image'  # image or ram
# ACTION_NUM = 2
# FRAME_SKIP = 4
# EPOCH_NUM = 360
# EPOSIDES_IN_EPOCH = 300
#
# PHI_LENGTH = 4
# CHANNEL = 3
# WIDTH = 160
# HEIGHT = 210
#
# DISCOUNT = 0.99
# RANDOM_SEED = int(time.time() * 1000) % 100000000
#
#
# LEARNING_RATE = 0.0025
# WEIGHT_DECAY = 0.0
#
# GRAD_CLIPPING_THETA = 0.01
#
# # reward
# DEATH_REWARD = -1
# POSITIVE_REWARD = 1
# NEGATIVE_REWARD = -1
#
# MODEL_PATH = '/home/hongruying/hellor_reply_priority/model'
# # MODEL_PATH = 'D:\\software_data\\seekloud\\hellorl\\model'
# MODEL_FILE_MARK = 'pong_pg_'
# BEGIN_TIME = time.strftime("%Y%m%d_%H%M%S")
#
# print('\n\n\n\n++++++++++++++++ edited time: 22:19 ++++++++++++++++++')
