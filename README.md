# About

# Environment

# Part 1
小车的资助驾驶采用行为克隆的方式，有数据采集、数据集预处理、模型训练、小车资助驾驶四个步骤。

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

