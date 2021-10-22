'''
在运行Auto_Driver.py之前可以使用该程序进行目标识别的测试
更新1：
    加入读取得分最高的标记的功能，并显示
'''
import sys
import time
import paddlex as pdx
import cv2

# 加载模型
predictor = pdx.deploy.Predictor('inference_model/inference_model_prune/inference_model')
result_ = predictor.predict(img_file='data_obj/images/1_cancel_10_1.jpg',warmup_iters=100, repeats=100)
pdx.det.visualize('data_obj/images/1_cancel_10_1.jpg', result_, save_dir='./')
#model = pdx.load_model('model/PP_YOLO_Tiny/TrafficSigns/best_model')
cap  = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret == True:
        time_begin = time.time()
        #frame = cv2.resize(frame, (320, 240))
        result = predictor.predict(frame)
        time_cost = time.time() - time_begin    # 时间代价
        frequency = round((1 / time_cost), 3)       # 预测频率
        img = pdx.det.visualize(frame, result, threshold=0.5, save_dir=None)
        text = 'frequency: ' + str(frequency) + 'Hz'
        img = cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 1)
        cv2.imshow('img', img)
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            cap.release()
            print('退出')
            sys.exit(0)
        
