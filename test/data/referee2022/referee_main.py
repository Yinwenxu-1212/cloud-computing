# -*- coding:utf-8-*-
import datetime
import json
import os
import sys
import time
import subprocess

import logger_common
from clean import clean_up
from compiler import compile_sdk
from error_code import RefereeExcpt
from predefs import TIME_DELTA
from scheduler import Scheduler
from solution_final import SolutionFinal
from solution_inter import SolutionInter
from solution_pre import SolutionPre
from predefs import DATASET_PATH
from predefs import COMMON_TIMEOUT
import random

LOGGER = logger_common.init_log("referee.log", "./")

def handling_runner_err(err_msg):
    res_dict = {"status": "fail",
                "cost": 0,
                "exec_time": 0,
                "except": "程序编译异常"}

    res_dict["except"] = str(err_msg)
    with open("./result.json", "w", encoding='utf8') as res_fd:
        res_fd.write(json.dumps(res_dict, ensure_ascii=False))
    subprocess.run(['chmod', '-R', 'go-rwx', DATASET_PATH], cwd='/', shell=False, timeout=COMMON_TIMEOUT)
    clean_up('/output/*')
    os._exit(0)


def handling_runner_success(cost, exec_time):
    res_dict = {"status": "success",
                "cost": 0,
                "exec_time": 0,
                "except": "程序运行成功"}
    res_dict["cost"] = cost
    res_dict["exec_time"] = exec_time
    with open("./result.json", "w", encoding='utf8') as res_fd:
        res_fd.write(json.dumps(res_dict, ensure_ascii=False))
    subprocess.run(['chmod', '-R', 'go-rwx', DATASET_PATH], cwd='/', shell=False, timeout=COMMON_TIMEOUT)
    clean_up('/output/*')
    os._exit(0)

def get_solution(stage):
    if stage == "0":
        solution = SolutionPre()
    elif stage == "1":
        solution = SolutionPre()
    elif stage == "2":
        solution = SolutionInter()
    elif stage == "3":
        solution = SolutionInter()
    elif stage == "4":
        solution = SolutionFinal()
    elif stage == "5":
        solution = SolutionFinal()
    return solution

def get_dataset(dataset_path):
    dataset_list = list()
    dirs = os.listdir(dataset_path)
    for dir in dirs:
        dataset_list.append(os.path.join(dataset_path, dir))
    return dataset_list

def prepare_data(data):
    # remove all the permission
    subprocess.run(['chmod', '-R' , 'go-rwx', DATASET_PATH], cwd='/', shell=False, timeout=COMMON_TIMEOUT)
    subprocess.run(['unlink', '/data'], cwd='/', shell=False, timeout=COMMON_TIMEOUT)
    subprocess.run(['ln', '-s', data, '/data'], cwd='/', shell=False, timeout=COMMON_TIMEOUT)
    subprocess.run(['chmod', 'o+x', DATASET_PATH], cwd='/', shell=False, timeout=COMMON_TIMEOUT)
    subprocess.run(['chmod', '-R', 'o+rx', data], cwd='/', shell=False, timeout=COMMON_TIMEOUT)

def compile_user_code(lang):
    start_time = (datetime.datetime.now() + datetime.timedelta(hours=TIME_DELTA)).strftime('%Y-%m-%d %H:%M:%S')
    LOGGER.info("start compile:" + str(start_time))
    try:
        compile_sdk(lang)
    except RefereeExcpt as err_cmp:
        LOGGER.info("use code compile failed:")
        handling_runner_err(err_cmp)
    except Exception as err:
        LOGGER.info("use code compile failed:" + str(err))
        handling_runner_err("选手程序编译失败")
    end_time = (datetime.datetime.now() + datetime.timedelta(hours=TIME_DELTA)).strftime('%Y-%m-%d %H:%M:%S')
    LOGGER.info("end compile:" + str(end_time))

def run_user_code(sdk):
    start_time = (datetime.datetime.now() + datetime.timedelta(hours=TIME_DELTA)).strftime('%Y-%m-%d %H:%M:%S')
    LOGGER.info("start run user code :" + str(start_time))
    exec_time = None
    try:
        # 运行选手程序
        sch = Scheduler()
        start = time.perf_counter()
        sch.run_student_code(sdk)
        exec_time = int((time.perf_counter() - start) * 1000)
    except RefereeExcpt as err_msg:
        end_time = (datetime.datetime.now() + datetime.timedelta(hours=TIME_DELTA)).strftime('%Y-%m-%d %H:%M:%S')
        LOGGER.error("end run user code, RefereeExcpt: " + str(err_msg) + str(end_time))
        handling_runner_err(err_msg)
    except Exception as err_msg:
        end_time = (datetime.datetime.now() + datetime.timedelta(hours=TIME_DELTA)).strftime('%Y-%m-%d %H:%M:%S')
        LOGGER.error("end run user code:" + str(err_msg) + str(end_time))
        handling_runner_err("选手程序运行失败")
    end_time = (datetime.datetime.now() + datetime.timedelta(hours=TIME_DELTA)).strftime('%Y-%m-%d %H:%M:%S')
    LOGGER.info("end run user code:" + str(end_time))
    return exec_time

def calc_cost(stage):
    solution = get_solution(stage)
    try:
        cost = solution.calc_cost()
    except RefereeExcpt as err_calc:
        LOGGER.error("calc cost failed :" + str(err_calc))
        handling_runner_err(err_calc)
    except Exception as err_msg:
        LOGGER.error("calc cost failed :" + str(err_msg))
        handling_runner_err(err_msg)
    return cost

def main(argv):
    """
    python3 referee.py 0/1/2/3/4/5 Python/PyPy/C/Cpp/Java
    语言类型：
    Python/PyPy -> ./CodeCraft-2022/src/CodeCraft-2022.py
    C/Cpp -> ./bin/CodeCraft-2022
    Java -> ./bin/CodeCraft-2022.jar
    比赛阶段：
    0：初赛练习赛
    1：初赛正式赛
    2：复赛练习赛
    3：复赛正式赛
    4：决赛练习赛
    5：决赛正式赛
    :param argv:
    :return:
    """
    dev_type = {"Python": ["python", "./CodeCraft-2022/src/CodeCraft-2022.py"],
                "PyPy": ["/usr/local/bin/pypy", "./CodeCraft-2022/src/CodeCraft-2022.py"],
                "C": "./bin/CodeCraft-2022",
                "Cpp": "./bin/CodeCraft-2022",
                "Java": ["/usr/bin/java", "-Xms64m", "-Xmx8g", "-Djava.library.path=./bin", "-classpath",
                         "./bin/CodeCraft-2022.jar", "com.huawei.java.main.Main", "2>&1"]}

    language = argv[1]
    sdk = dev_type[language]
    LOGGER.info("student language: " + language)

    # compile
    compile_user_code(language)

    # get total datasets
    total_cost = 0
    total_time = 0

    datasets = get_dataset(DATASET_PATH)
    random.shuffle(datasets)

    for data in datasets:
        prepare_data(data)
        #run user code
        exec_time = run_user_code(sdk)
        # calc cost
        cost = calc_cost(argv[0])

        total_cost += cost
        total_time += exec_time

    # output result
    LOGGER.info("run success")
    handling_runner_success(total_cost, total_time)


if __name__ == '__main__':
    clean_up('/output/*')
    start_time = (datetime.datetime.now() + datetime.timedelta(hours=TIME_DELTA)).strftime('%Y-%m-%d %H:%M:%S')
    LOGGER.info("start time:" + str(start_time))
    main(sys.argv[1:])
    end_time = (datetime.datetime.now() + datetime.timedelta(hours=TIME_DELTA)).strftime('%Y-%m-%d %H:%M:%S')
    LOGGER.info("end time:" + str(end_time))
    clean_up('/output/*')
