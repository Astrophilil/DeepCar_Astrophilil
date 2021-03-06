# About

# Environment

# Part 1
小车的自主驾驶采用行为克隆的方式，有数据采集、数据集预处理、模型训练、小车资助驾驶四个步骤。

## 车道线数据采集
1. 检测键盘、鼠标和手柄的USB接收器是否插入
2. 终端使用cd命令切换到./src/目录下
3. 输入命令python3 Data_Coll.py开始运行
4. 开启手柄，按B键启动小车，并控制小车开始采集数据

## 数据集预处理
1. 完成车道线数据采集之后，会生成一个data文件夹
2. 终端cd到src文件夹下运行Img_Handle.py，预处理后的数据保存在data/hsv_img
3. 运行Create_Data_Liet.py，将数据集分成训练集和验证集，在data下的train.list和test.list

## 训练
1. 本地训练，运行python Train_Model.py，模型保存在data/model_infer/
2. 线上训练 https://aistudio.baidu.com/aistudio/index

## 小车自主驾驶
1. 在那之前可以使用Driver_self_use_data.py进行测试
2. 直接运行Driver_self.py

# Part 2
小车在识别标志物的过程中完成自主驾驶，包括数据采集、数据标注、训练、部署

## 标志物数据采集
1. 根据需求更改/src/文件夹中的get_imgs_for_obj_detection.py
2. 运行该程序进行采集

## 打标签
1. 使用labelImg工具，以VOC格式保存

## 训练
1. 模型使用了PP-YOLO Tiny，直接使用paddlex进行模型的构建和训练
2. 模型裁剪
3. 模型量化

## 模型测试
1. 将训练好的模型保存之model的相应文件夹中
2. 运行obj_detection_test.py

## 自主驾驶
1. 拷贝相应的模型文件到本地
2. 测试识别效果，运行obj_detection_test.py
3. 自主驾驶