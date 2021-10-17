'''
用于拍摄相关的数据集
运行前需要选择拍摄的标志物的label_name数组中的index
还有需要采集的数据的数量
'''

import cv2
import time
import sys

cap = cv2.VideoCapture(0)
t = int(time.time())
label_name = [
    'cancel_10',
    'crossing',
    'limit_10',
    'green',
    'red',
    'straight',
    'turn_left',
    'turn_right',
    'ramp',
    'traffic_lights'
]

'''------------运行前需要指定----------------'''
index_obj = 9
num_img = 80

print(sys.path[0])
print("注意在运行之前选择index_obj：")
for i in range(len(label_name)):
    print('{}:{}'.format(i, label_name[i]))

index = 0
while True:
    ret, frame = cap.read()
    if ret == True:
        # cv2.imwrite('lx_data/images/' + str(t) + '_' + str(num) + '.jpg', frame)
        img_name = "{}/../data_obj/images/2_{}_{}.jpg".format(sys.path[0], label_name[index_obj], str(index))
        print("保存图片：", img_name)
        cv2.imwrite(img_name, frame)
        index += 1
        cv2.imshow('sign', frame)
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            cap.release()
            break
        time.sleep(1)
        if index == num_img:
            break

print("注意在运行之前选择index_obj：")
for i in range(len(label_name)):
    print('{}:{}'.format(i, label_name[i]))