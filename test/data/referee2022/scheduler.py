import os
import signal
import subprocess

import logger_common
import popen
from error_code import RefereeExcpt
from predefs import PLAYER_RUN_TIMEOUT

LOGGER = logger_common.init_log("referee.log", "./")

class Scheduler:
    """
    所有比赛阶段调度类的基础类
    """

    def __init__(self):
        subp_id = None

    def kill_student_proc(self):
        try:
            #os.killpg(self.subp_id, signal.SIGKILL)
            os.killpg(os.getpgid(self.subp_id), signal.SIGKILL)
        except:
            pass

    def run_student_code(self, sdk):
        subp = popen.call(sdk, subprocess.DEVNULL, subprocess.PIPE)
        self.subp_id = subp.pid
        out = bytes()
        try:
            out, err = subp.communicate(timeout=PLAYER_RUN_TIMEOUT)
        except subprocess.TimeoutExpired as err_msg:
            LOGGER.error("student code timeout")
            self.kill_student_proc()
            raise RefereeExcpt(2)
        except Exception as error:
            LOGGER.error(str(error))
            self.kill_student_proc()
            raise RefereeExcpt(15)
        # LOGGER.info("aha :" + out.decode())
        if subp.returncode != 0:
            LOGGER.error(str(err))
            LOGGER.error("user returned error: " + os.strerror(subp.returncode))
            raise RefereeExcpt(15)
