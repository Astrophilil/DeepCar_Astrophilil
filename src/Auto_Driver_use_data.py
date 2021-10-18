'''
该程序用于最终自主驾驶的测试，车道线的采集并为使用摄像头，而是用data
使用了paddlex框架的object detection
'''
from ctypes import *
import numpy as np
import cv2

import paddle.fluid as fluid
from PIL import Image
import paddle
import paddlex as pdx   

paddle.enable_static()
import sys,os
#import torch

#from models.experimental import attempt_load
#from utils.datasets import letterbox
#from utils.general import (non_max_suppression)
#from utils.torch_utils import select_device
from multiprocessing import Process, Queue
import time

# 速度
vel = 1550
# 转向角
angle = 1500
q = Queue()



def lane():
    global vel
    global angle

    def dataset(frame):
        lower_hsv = np.array([26, 43, 46])
        upper_hsv = np.array([34, 255, 255])

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)

        img = Image.fromarray(mask)
        img = img.resize((120, 120), Image.ANTIALIAS)
        img = np.array(img).astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = img.transpose((2, 0, 1))
        img = img[(2, 1, 0), :, :] / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    # 加载模型
    save_path = "/home/a/DeepCarQing/ArtRobot_DeepCar/model/model_infer/"
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    [infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(dirname=save_path, executor=exe)
    #lib_path = "../lib/libart_driver.so"
    lib_path = "/home/a/DeepCarQing/ArtRobot_DeepCar/lib/libart_driver.so"
    so = cdll.LoadLibrary
    lib = so(lib_path)
    # car = "/dev/ttyACM0"
    car = '/dev/ttyUSB0'
    lib.art_racecar_init(38400, car.encode("utf-8"))
    '''============ lx ============='''
    #img_path = '/home/a/DeepCarQing/ArtRobot_DeepCar/data/hsv_img/'
    img_path = '/home/a/DeepCarQing/ArtRobot_DeepCar/data0/img/'
    try:
        if (lib.art_racecar_init(38400, car.encode("utf-8")) < 0):
            raise
            pass
       # if (lib.art_racecar_init(38400, car.encode("utf-8")) < 0):
        #    raise
        #    pass
        #os.system("gnome-terminal -e 'bash -c\"python drive.py; exec bash\"'")
        #os.system("gnome-terminal -e 'bash -c\"python drive.py; exec bash\"'")
    
        # lib.art_racecar_init(38400, car.encode("utf-8"))
        # lib.send_cmd(vel, angle)
        #cap = cv2.VideoCapture('/dev/cam_lane')
        #cap从cv2读取图片，而非视频
        i = 0
        while True:
            img_name = img_path+'{}.jpg'.format(i)
            i = i + 1
            print(img_name)
            frame = cv2.imread(img_name, flags=1)
            font = cv2.FONT_HERSHEY_DUPLEX # 设置字体
            #ret, frame = cap.read()
            if i:
                img = dataset(frame)
                result = exe.run(program=infer_program, feed={feeded_var_names[0]: img}, fetch_list=target_var)
                print(result)
                angle = result[0][0][0]
                angle = int(angle)
                print('转向角=', angle)
                # if 1400 <= angle <= 1600:
                #     vel = 1550
                # else:
                #     vel = 1540
                # if not q.empty():
                #     vel = q.get()
                #     print(vel)
                lib.send_cmd(vel, angle)
                result_value = 'result:({}, {})'.format(vel, angle)
                # 图片对象、文本、像素、字体、字体大小、颜色、字体粗细
                frame = cv2.putText(frame, result_value, (100, 100), font, 0.75, (0,0,0), 1,)
                cv2.imshow('frame', frame)
                time.sleep(0.03)
                if cv2.waitKey(1) == 27:
                    #lib.send_cmd(1500, 1500)
                    cv2.destroyAllWindows()
                    #cap.release()
                    sys.exit(0)
                    break
            else:
                print('lane相机打不开')
    except:
        pass
    finally:
        #lib.send_cmd(1500, 1500)
        pass

def sign_PPYoloTiny():
    global vel
    global angle
    model = pdx.load_model('model/PP_YOLO_Tiny/TrafficSigns/best_model')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if ret == True:
            time_begin = time.time()
            result = model.predict(frame) # 预测
            # 取最高得分的label
            for id in result:
                pass
            # 针对最高得分的进行相关的标记

            # 显示

def sign():
    global vel
    global angle
    device = select_device('cpu')
    half = device.type != 'cpu'  # half precision only supported on CUDA
    weights = '../model/yolov5_model/best.pt'
    # Load model

    # 加载模型
    model = attempt_load(weights, map_location=device)  # load FP32 model
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    cap = cv2.VideoCapture('/dev/cam_sign')
    print('打开相机')

    while True:
        ret, image = cap.read()
        if ret == True:
            # image = cv2.resize(image, (120, 120))
            with torch.no_grad():
                img = letterbox(image, new_shape=160)[0]
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                pred = model(img, augment=False)[0]
                pred = non_max_suppression(pred, 0.4, 0.5, classes=False, agnostic=False)
                coor = []
                f = pred[0].tolist()
                # print(len(f))
                # if len(f) == 0:
                #     if not q.empty():
                #         print(q.get())
                #         if q.get() == 1525:
                #              q.put(1550)   
                # else:
                for i, det in enumerate(pred):
                    for *xyxy, conf, cls in reversed(det):
                        for i in xyxy:
                            i = i.tolist()
                            i = int(i)
                            label = names[int(cls)]
                            conf = round(float(conf), 2)
                            # label = '%s %.2f' % (names[int(cls)], conf)
                            # print(conf)
                            if conf >= 0.5:
                                coor.append(i)
                                label = str(label)
                                # print(label)
                                if label == 'cancel_10':
                                    # 限速解除
                                    # print('限速解除')
                                    vel = 1550
                                    q.put(vel)
                                elif label == 'crossing':
                                    # print('人行道')
                                    vel = 1525
                                    q.put(vel)
                                elif label == 'limit_10':
                                    # print('限速')
                                    vel = 1535
                                    q.put(vel)
                                elif label == 'strsight':
                                    # print('直行')
                                    pass
                                elif label == 'turn_left':
                                    # print('左转')
                                    pass
                                elif label == 'turn_right':
                                    # print('右转')
                                    pass
                                elif label == 'paper_red':
                                    # print('红灯')
                                    vel = 1500
                                    q.put(vel)
                                elif label == 'paper_green':
                                    # print('绿灯')
                                    vel = 1550
                                    q.put(vel)
                            # else:
                            #     vel = 1550
                            #     q.put(vel)
                    num = len(coor)
                    if num == 4:
                        cv2.rectangle(image, (coor[0] * 4, coor[1] * 4), (coor[2] * 4, coor[3] * 4), (0, 255, 0), 7)
                        # x = int((coor[2] - coor[0]) / 2 + coor[0])
                        # y = int((coor[3] - coor[1]) / 2 + coor[1])
                        # cv2.rectangle(image, (coor[0], coor[1]), (coor[2], coor[3]), (0, 255, 0), 7)

            cv2.imshow('sign', image)
            k = cv2.waitKey(1)
            if k == 27:
                cv2.destroyAllWindows()
                cap.release()
                sys.exit(0)
                break
        else:
            print('sign相机打不开')


if __name__ == '__main__':
    lane_run = Process(target=lane)
    sign_run = Process(target=sign_PPYoloTiny)

    lane_run.start()
    sign_run.start()
