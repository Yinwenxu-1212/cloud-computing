import json
import subprocess
import sys

if __name__ == '__main__':
    command = json.loads(sys.argv[1])
    p = subprocess.Popen(command, cwd='./')
    p.communicate()
    sys.exit(p.poll())
