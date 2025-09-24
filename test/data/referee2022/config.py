import configparser
from predefs import CONFIG_FILE
def read_config(section, name):
    cfg = configparser.ConfigParser()
    cfg.read(CONFIG_FILE)
    qos = cfg.get(section, name)
    return qos

if __name__ == '__main__':
    qos = read_config('config', 'qos_constraint')
    print(int(qos))
