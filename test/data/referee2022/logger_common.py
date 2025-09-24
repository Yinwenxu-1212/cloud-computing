# endcoding:utf-8
"""
logger module
"""
import logging
import logging.config
import logging.handlers
import os
import stat
import zipfile
from functools import wraps
from inspect import signature
from logging.handlers import RotatingFileHandler, WatchedFileHandler

CRITICAL = 50
FATAL = CRITICAL
ERROR = 40
WARNING = 30
WARN = WARNING
INFO = 20
DEBUG = 10
NOTSET = 0
# LOG_FORMAT = "%(asctime)s|%(filename)s:%(lineno)d|%(funcName)s:%(message)s"
LOG_FORMAT = "%(message)s"
DATE_FORMAT = '%Y-%m-%d  %H:%M:%S %a'
LOG_LEVEL = DEBUG
MAX_BYTES = 10 * 1024 * 1024
BACKUP_COUNT = 2


class LogHandler(RotatingFileHandler, WatchedFileHandler):
    def __init__(self, comp_method="ZIP", *args, **kwargs):
        RotatingFileHandler.__init__(self, *args, **kwargs)
        self.namer, self.rotator = self._get_namer_and_rotator(comp_method)
        self.dev, self.ino = -1, -1
        self._statstream()

    def emit(self, record):
        self.reopenIfNeeded()
        RotatingFileHandler.emit(self, record)

    def _get_namer_and_rotator(self, comp_method):
        namer_str = "namer_%s" % (comp_method.lower())
        rotator_str = "rotator_%s" % (comp_method.lower())
        try:
            namer = getattr(self, namer_str)
            rotator = getattr(self, rotator_str)
        except (AttributeError, NameError, TypeError):
            namer = self.namer_zip
            rotator = self.rotator_zip
        return namer, rotator

    @staticmethod
    def namer_zip(default_name):
        return default_name + ".zip"

    @staticmethod
    def rotator_zip(src, dst):
        """
        :param src:
        :param dst:
        :return:
        """
        _, file_name = os.path.split(src)
        with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as file:
            file.write(src, arcname=file_name)

        if os.path.isfile(src) and os.path.isfile(dst):
            os.remove(src)
        os.chmod(dst, stat.S_IREAD + stat.S_IWRITE)

    def doRollover(self):
        RotatingFileHandler.doRollover(self)
        os.chmod(self.baseFilename, stat.S_IREAD + stat.S_IWRITE)


class AntiCRLFLogRecord(logging.LogRecord):
    """
    记录日志时，转义CRLF
    """

    def getMessage(self):
        """
        重写getMessage方法，以转义CRLF
       '\n' --> '\\n'
       '\r' --> '\\r'
        """
        message = str(self.msg)
        if self.args:
            message = message % self.args
        message = message.replace('\n', '\\n').replace('\r', '\\r')
        return message


def init_log(file_name, base_log_dir, log_flag=""):
    """
    Log function
    :param file_name:
    :param base_log_dir:
    :param log_flag:
    :return:
    """
    logging.setLogRecordFactory(AntiCRLFLogRecord)
    logger_name = file_name.split(".")[0]
    if not os.path.isdir(base_log_dir):
        os.makedirs(base_log_dir)
        os.chmod(base_log_dir, stat.S_IRWXU)
    # 如果调用函数指定了log_flag，则logger_name根据log_flag+日志文件名组合，
    # 防止不同任务日志重名导致日志丢失的问题
    if log_flag:
        logger_name = "{}-{}".format(logger_name, log_flag)
    log_dir = os.path.join(base_log_dir, file_name)

    handler = LogHandler(filename=log_dir, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT)
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    handler.setFormatter(formatter)
    logger = logging.getLogger(logger_name)
    logger.setLevel(LOG_LEVEL)

    if not logger.handlers:
        logger.addHandler(handler)
    os.chmod(log_dir, stat.S_IREAD + stat.S_IWRITE)
    return logger


def fun_par_log(fun_ret_type=None, *args, **kwargs):
    """
    装饰器，记录行数调用状态，并对传入参数的数据类型的合法性进行 强制 判断，参数数据类型错误时报错
    :param fun_ret_type:约束装饰func的返回值类型
    :param args:装饰func的传入参数（非指定情况）进行数据类型限制，对迭代数据可以限制成(set,list,tuple)
    :param kwargs:装饰func的传入参数（指定情况）进行数据类型限制，对迭代数据可以限制成(set,list,tuple)
    :return: 返回被装饰行数的返回值
    """
    main_log = init_log("Mainlog.log", "./")

    def decorator(fun):
        # 获取被装饰函数的传入参数
        sig = signature(fun)
        # 获取 装饰器需要判断的参数类型
        btypes = sig.bind_partial(*args, **kwargs).arguments

        @wraps(fun)
        def wrapper(*funargs, **funkwargs):
            # log_string = fun.__name__ + " 被调用"
            log_string = fun.__name__ + " was called..."
            main_log.info(log_string)
            # 获取 被装饰函数需要判断的参数类型
            fun_btypes = sig.bind_partial(*funargs, **funkwargs).arguments
            for fun_par_name, fun_par_value in fun_btypes.items():
                if fun_par_name in btypes:
                    if not isinstance(fun_par_value, btypes[fun_par_name]):
                        main_log.info("parameter TypeError, function: %s, parameter:%s"
                                      "The valid type of the parameter is %s , but now is %s." % (
                                          fun.__name__, fun_par_name, btypes[fun_par_name], type(fun_par_value)))
                        # main_log.info("调用函数：%s 时传入参数的数据类型出错，参数名：%s 的合法数据类型：%s 当前数据类型：%s" % (
                        #     fun.__name__, fun_par_name, btypes[fun_par_name], type(fun_par_value)))
                        raise TypeError("'%s' must be '%s'" % (fun_par_name, btypes[fun_par_name]))
            try:
                # 对函数返回值的数据类型进行判断
                fun_ret = fun(*funargs, **funkwargs)
            except Exception:
                main_log.info("parameter TypeError, function: %s, parameter:%s"
                              "The valid type of the parameter is %s , but now is %s." % (
                                  fun.__name__, fun_par_name, btypes[fun_par_name], type(fun_par_value)))
                # main_log.exception("调用函数：%s 出错，参数名：%s 的合法数据类型：%s 当前数据类型：%s" % (
                #     fun.__name__, fun_par_name, btypes[fun_par_name], type(fun_par_value)))
            if fun_ret_type is not None:
                if not isinstance(fun_ret, fun_ret_type):
                    main_log.info("return parameter TypeError, function: %s, parameter:%s"
                                  "The valid type of the parameter is %s , but now is %s." % (
                                      fun.__name__, fun_par_name, btypes[fun_par_name], type(fun_par_value)))
                    # main_log.info("调用函数：%s 时返回值的数据类型出错，函数返回值的合法数据类型：%s 当前数据类型：%s" % (
                    #     fun.__name__, fun_ret_type, type(fun_ret)))
            return fun_ret

        return wrapper

    return decorator


def log_calling_init():
    """
    初始化学生程序日志，及判题运行日志对象。
    stu_logger: 学生日志
    sch_logger: 调度器运行日志
    :return: 返回两个对象
    """
    stu_logger = init_log("run_student.log", "./")
    sch_logger = init_log("run_scheduler.log", "./")
    return stu_logger, sch_logger


if __name__ == '__main__':
    # 调试
    main_log = init_log("Mainlog.log", "./")


    @fun_par_log(fun_ret_type=dict, a=str, b=dict)
    def test_log(a, b, c=0):
        print('input a:', a, type(a))
        print('input b:', b, type(b))
        return a, b, c


    b = {'name': 'runoob', 'likes': 123, 'url': 'www.runoob.com'}
    test_log('aaaa', b, )
