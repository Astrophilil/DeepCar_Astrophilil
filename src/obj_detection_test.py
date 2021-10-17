'''
在运行Auto_Driver.py之前可以使用该程序进行目标识别的测试
'''
import sys
import time
import paddlex as pdx
import cv2

# 加载模型
model = pdx.load_model('model/PP-YOLO Tiny/Fall/best_model')
cap  = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret == True:
        time_begin = time.time()
        #frame = cv2.resize(frame, (320, 240))
        result = model.predict(frame)
        time_cost = time.time() - time_begin    # 时间代价
        frequency = round((1 / time_cost), 3)       # 预测频率
        img = pdx.det.visualize(frame, result, threshold=0.5, save_dir=None)
        text = 'frequency: ' + str(frequency) + 'Hz'
        img = cv2.putText(img, text, (100, 100), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0,0,0), 1)
        cv2.imshow('img', img)
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            cap.release()
            print('退出')
            sys.exit(0)
        
