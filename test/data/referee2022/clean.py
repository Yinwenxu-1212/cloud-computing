import subprocess

# 清除用户残留数据
def clean_up(file_path):
    try:
        ret = subprocess.run('rm -rf '+ file_path, cwd='./', shell=True, timeout=100)
    except Exception as error:
        pass
