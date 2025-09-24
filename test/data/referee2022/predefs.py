# 31天总共8928个计费点
# 31升序排列后95%计费点为第8482计费点，索引从0开始，索引值为8481
# 降序排列95%计费点为446，好处是不用考虑没有流量的客户
# 30天总共8640个计费点
# 31升序排列后95%计费点为第8208计费点，索引从0开始，索引值为8207
# 降序排列95%计费点为432，好处是不用考虑没有流量的客户
COMPILE_TIMOUT = 30  # 编译超时30s
COMMON_TIMEOUT = 20
PLAYER_RUN_TIMEOUT = 300  # 选手答题超时300s
TIME_DELTA = 8  # 时区偏移，docker里面的时区是0
# PRECISION = 4  # 成本保留4位小数

DATASET_PATH = '/dataset'
UNITED_STREAM_FILE = '/data/demand.csv'
DETAILED_STREAM_FILE = '/data/demand.csv'
QOS_FILE = '/data/qos.csv'
BANDWITH_FILE = '/data/site_bandwidth.csv'
CONFIG_FILE = '/data/config.ini'
SOLUTION_FILE = '/output/solution.txt'

# DATASET_PATH = '../dataset'
# DATA_PREFIX = '../data/'
# DETAILED_STREAM_FILE = DATA_PREFIX + '/demand.csv'
# UNITED_STREAM_FILE = DATA_PREFIX + '/demand.csv'
# QOS_FILE = DATA_PREFIX + 'qos.csv'
# BANDWITH_FILE = DATA_PREFIX + 'site_bandwidth.csv'
# CONFIG_FILE = DATA_PREFIX + 'config.ini'
# SOLUTION_FILE = DATA_PREFIX + 'solution.txt'

