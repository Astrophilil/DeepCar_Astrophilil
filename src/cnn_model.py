'''
Paddle模型函数
'''
import paddle.fluid as fluid

def cnn_model(image):
    conv1 = fluid.layers.conv2d(input=image, num_filters=24, filter_size=(5, 5), stride=(2, 2), act='relu')
    pool0 = fluid.layers.pool2d(input=conv1, pool_size=(2, 2))
    conv2 = fluid.layers.conv2d(input=pool0, num_filters=36, filter_size=(5, 5), stride=(2, 2), act='relu')
    pool1 = fluid.layers.pool2d(input=conv2, pool_size=(2, 2))
    conv3 = fluid.layers.conv2d(input=pool1, num_filters=48, filter_size=(5, 5), stride=(2, 2), act='relu')
    pool2 = fluid.layers.pool2d(input=conv3, pool_size=(2, 2))
    conv4 = fluid.layers.conv2d(input=pool2, num_filters=64, filter_size=(3, 3), act='relu')
    pool3 = fluid.layers.pool2d(input=conv4, pool_size=(2, 2))
    drop1 = fluid.layers.dropout(pool3, dropout_prob=0.2)
    conv5 = fluid.layers.conv2d(input=drop1, num_filters=64, filter_size=(3, 3), act='relu')
    fla = fluid.layers.flatten(conv5)
    drop2 = fluid.layers.dropout(fla, dropout_prob=0.2)
    fc1 = fluid.layers.fc(input=drop2, size=100, act='relu')
    fc2 = fluid.layers.fc(input=fc1, size=50, act='relu')
    fc3 = fluid.layers.fc(input=fc2, size=10, act='relu')
    predict = fluid.layers.fc(input=fc3, size=1)
    return predict