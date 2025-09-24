import os
import signal
import subprocess

import logger_common
import popen
from error_code import RefereeExcpt
from predefs import COMPILE_TIMOUT
from predefs import COMMON_TIMEOUT

LOGGER = logger_common.init_log("referee.log", "./")


def kill_compiler(pid):
    try:
        # os.killpg(self.subp_id, signal.SIGKILL)
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except:
        pass


def compile_sdk(language):
    if language in ["Cpp", "C"]:
        subprocess.run(["cp", "./SDK/C/build.sh", "../student/"], cwd='./', shell=False, timeout=COMMON_TIMEOUT)
    elif language == "Java":
        subprocess.run(["cp", "./SDK/Java/build.sh", "../student/"], cwd='./', shell=False, timeout=COMMON_TIMEOUT)

    subprocess.run(["cp", "./worker.py", "../student/"], cwd='./', shell=False, timeout=COMMON_TIMEOUT)

    subprocess.run(["chown", "-R", "student" + ":students", "../student/"], cwd='./', shell=False)
    subprocess.run(["chmod", "755", "../"], cwd='./', shell=False, timeout=COMMON_TIMEOUT)
    subprocess.run(["chmod", "755", "../student"], shell=False, timeout=COMMON_TIMEOUT)
    subprocess.run(["chmod", "-R", "700", "../student/"], cwd='./', shell=False, timeout=COMMON_TIMEOUT)
    if language in ["Python", "PyPy"]:
        subprocess.run(["chmod", "-R", "500", "../student/"], cwd='./', shell=False, timeout=COMMON_TIMEOUT)
        return

    # subp = subprocess.Popen("sudo -u student sh build.sh", cwd='../student', shell=True, close_fds=True, preexec_fn = os.setsid,
    #                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    subp = popen.call(["sh", "build.sh"], subprocess.PIPE, subprocess.STDOUT)
    out = bytes()
    try:
        out, err = subp.communicate(timeout=COMPILE_TIMOUT)
    except subprocess.TimeoutExpired as err_msg:
        # LOGGER.error(str(err_msg))
        kill_compiler(subp.pid)
        raise RefereeExcpt(1)
    except Exception as error:
        kill_compiler(subp.pid)
        raise RefereeExcpt(0)

    if "make return: 0" in out.decode() or "build jar success" in out.decode():
        if language in ["Cpp", "C"]:
            subprocess.run(["chmod", "-R", "500", "../student/"], cwd='./', shell=False)
        elif language == "Java":
            subprocess.run(["chmod", "-R", "500", "../student/"], cwd='./', shell=False)
    else:
        LOGGER.error(out.decode())
        raise RefereeExcpt(0)


if __name__ == '__main__':
    # compile_sdk("Cpp")
    # compile_sdk("Python")
    # compile_sdk("PyPy")
    compile_sdk("Java")
