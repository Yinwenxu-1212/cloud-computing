# -*- coding:utf-8-*-
import sys

import logger_common

LOGGER = logger_common.init_log("referee.log", "./")

error_dict = {
    0: "程序编译异常",
    1: "程序编译超时(30s)",
    2: "选手程序运行超时",
    3: "分配方案不合法",
    4: "分配方案不合法",
    5: "分配方案不合法",
    6: "分配方案不合法",
    7: "分配方案不合法",
    8: "分配方案不合法",
    9: "读取配置文件失败",
    10: "分配方案不合法",
    11: "未知错误",
    12: "打开选手输出文件失败",
    13: "分配方案不合法",
    14: "分配方案不合法",
    15: "选手程序运行失败",
    16: "数据集解析失败",
    17: "分配方案不合法"
}

error_dict_en = {
    0: "compile error",
    1: "compile timeout(30s)",
    2: "run timeout(300s)",
    3: "output format error",
    4: "customer streams not satisfy QoS",
    5: "some customers have not alloc streams in 5 min",
    6: "some customers alloced streams not equal required",
    7: "unkonwn edge node",
    8: "streams exceeds edge node bw",
    9: "streams exceeds center node bw",
    10: "stream is allocated many times for single customer",
    11: "unknown error",
    12: "open output file failed",
    13: "unkonw stream，streams are not from input dataset",
    14: "streams are not from edge node",
    15: "run student code error",
    16: "parse dataset failed",
    17: "negative stream value"
}

# 自定义异常
class RefereeExcpt(Exception):
    def __init__(self, err_code):
        self.err_code = err_code
        LOGGER.error("------------------------------------------")
        LOGGER.error("Raise Exception:"+error_dict_en.get(err_code, "unknown error: " + str(err_code)))
        LOGGER.error("file name:"+ sys._getframe(1).f_code.co_filename)  # 当前文件名，可以通过__file__获得
        LOGGER.error("func name:"+ str(sys._getframe(1).f_code.co_name))  # 调用该函数的函数名字，如果没有被调用，则返回<module>
        LOGGER.error("line number:"+ str(sys._getframe(1).f_lineno))  # 调用该函数的行号
        LOGGER.error("------------------------------------------")

    def __str__(self):
        return error_dict.get(self.err_code, "未知错误: "+ str(self.err_code))

def raise_excetp():
    try:
        raise RefereeExcpt(1)
    except RefereeExcpt as e:
        print(str(e))
    except Exception as err:
        print(str(err))


if __name__ == '__main__':
    raise_excetp()