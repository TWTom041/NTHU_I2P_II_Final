import subprocess
import time
import threading

proc = subprocess.Popen(["./build/minichess-ubgi.exe"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

def send(cmd):
    proc.stdin.write((cmd + "\n").encode())
    proc.stdin.flush()

def read_output():
    while True:
        line = proc.stdout.readline().decode().strip()
        if not line:
            break
        print("Engine:", line)

t = threading.Thread(target=read_output, daemon=True)
t.start()

send("ubgi")
send("setoption name Algorithm value submission")
send("isready")
time.sleep(1)
send("go movetime 10000")
time.sleep(11)
proc.kill()
