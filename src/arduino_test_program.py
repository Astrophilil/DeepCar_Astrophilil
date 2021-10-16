'''
用于测试arduino的程序
'''
from ctypes import *
#import serial
import os
import sys
import time

print(sys.path[0])

lib_path = sys.path[0] + "/../lib/libart_driver.so"
so = cdll.LoadLibrary
lib = so(lib_path)
car = "/dev/ttyUSB0"
# car = serial.value
#lib.art_racecar_init(38400, car.encode("utf-8"))

if (lib.art_racecar_init(38400, car.encode("utf-8")) < 0):
    raise
    pass
#os.system("gnome-terminal -e 'bash -c\"python drive.py; exec bash\"'")
#os.system("gnome-terminal -e 'bash -c\"python drive.py; exec bash\"'")
lib.send_cmd(1500, 1500)

print("加速")
data = [1530, 1500]
lib.send_cmd(data[0], data[1])
time.sleep(1)
data = [1550, 1500]
lib.send_cmd(data[0], data[1])
time.sleep(1)
data = [1580, 1500]
lib.send_cmd(data[0], data[1])
time.sleep(1)
data = [1600, 1500]
lib.send_cmd(data[0], data[1])
time.sleep(1)

data = [1550, 1500]
lib.send_cmd(data[0], data[1])
time.sleep(1)

print("右转")
data = [1550, 1400]
lib.send_cmd(data[0], data[1])
time.sleep(2)
data = [1550, 1300]
lib.send_cmd(data[0], data[1])
time.sleep(2)
data = [1550, 1200]
lib.send_cmd(data[0], data[1])
time.sleep(2)

data = [1550, 1500]
lib.send_cmd(data[0], data[1])
time.sleep(1)

print("左转")
data = [1550, 1600]
lib.send_cmd(data[0], data[1])
time.sleep(2)
data = [1550, 1700]
lib.send_cmd(data[0], data[1])
time.sleep(2)
data = [1550, 1800]
lib.send_cmd(data[0], data[1])
time.sleep(2)

data = [1550, 1500]
lib.send_cmd(data[0], data[1])
time.sleep(1)

print("停")
data = [1500, 1500]
lib.send_cmd(data[0], data[1])
time.sleep(1)
