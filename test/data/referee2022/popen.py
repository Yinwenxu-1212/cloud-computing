import json
import os
import subprocess


class Popen:

    def call(self, command, out, err):
        cmd = json.dumps(command)
        return self.run(cmd, out, err)

    def run(self, cmd, out, err):
        p = subprocess.Popen(
            "sudo -u student educloud /usr/bin/python worker.py '{}'".format(cmd),
            cwd='../student/', shell=True, close_fds=True, preexec_fn=os.setpgrp,
            stdout=out, stderr=err)
        return p


def call(command, out=subprocess.DEVNULL, err=subprocess.DEVNULL):
    """
    :param command: 要执行的命令行
    :return:
    """
    return Popen().call(command, out, err)
