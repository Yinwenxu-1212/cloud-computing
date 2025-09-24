import json
import subprocess
import sys

from clean import clean_up

if __name__ == '__main__':
    clean_up('./referee.log')
    clean_up('./result.json')
    command = ["python", "referee_main.py"] + sys.argv[1:]
    p = subprocess.Popen(command, cwd='./')
    p.communicate()
    if p.poll() != 0:
        res_dict = {"status": "fail",  # 选手程序执行状态
                    "cost": 0,  # 总成本
                    "exec_time": 0,  # 选手程序执行时间
                    "except": "选手程序运行失败"}  # 异常信息
        with open("./result.json", "w", encoding='utf8') as res_fd:
            res_fd.write(json.dumps(res_dict, ensure_ascii=False))
