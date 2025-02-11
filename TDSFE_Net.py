from keras.models import Model
from keras.layers import add, Layer, Input, Conv1D, Activation, Flatten, Dense, GRU, Lambda, Dropout, Concatenate, \
    SpatialDropout1D
from keras.utils import to_categorical
import scipy.io
import numpy as np
# from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.layers import SeparableConv1D, MaxPooling1D, AveragePooling1D, GlobalMaxPooling1D
from keras.layers import BatchNormalization
from keras import backend as K
from keras.optimizers import SGD, Nadam, Adam, RMSprop
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import tensorflow as tf
import keras
from keras import Sequential
from keras.src.applications.densenet import layers
from keras.src.layers import Bidirectional, DepthwiseConv1D, GlobalAveragePooling1D, Add, Reshape, Multiply
from matplotlib import pyplot as plt
import pandas as pd
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping


# Residual block 残差块
def TSCN(x, filters, kernel_size, dilation_rate, F2):
    a1 = Conv1D(filters, kernel_size, padding='same', kernel_initializer='he_uniform', dilation_rate=dilation_rate,
               activation='relu')(x)
    a2 = BatchNormalization()(a1)
    a3 = Activation('relu')(a2)
    a2 = SpatialDropout1D(0.2)(a3)
    a2 = SeparableConv1D(F2, kernel_size, strides=1, use_bias=False, padding='same', dilation_rate=1,
                        depth_multiplier=1)(a2)
    a2 = BatchNormalization()(a2)
    a2 = Activation('relu')(a2)
    a2 = SpatialDropout1D(0.2)(a2)
    if x.shape[-1] == filters:
        shortcut = x
    else:
        shortcut = Conv1D(filters, kernel_size, padding='same')(x)  # shortcut (shortcut)
    b = add([a2, shortcut])
    b = Activation('relu')(b)
    return b

def slice(x, index):
    return x[:, :, :, index]

class BiLSTM(keras.Model):
    def __init__(self, args):
        super(BiLSTM, self).__init__()
        self.lstm = Sequential()
        for i in range(args.num_layers):
            self.lstm.add(
                Bidirectional(layers.LSTM(units=args.hidden_size, input_shape=(args.seq_len, args.input_size),
                                          activation='tanh', return_sequences=True)))
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(args.output_size)

    def call(self, data, training=None, mask=None):
        x = self.lstm(data)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

class Args:
    def __init__(self):
        self.num_layers = 2  # LSTM层数
        self.hidden_size = 128  # LSTM隐藏层大小
        self.seq_len = 10  # 输入序列的长度
        self.input_size = 20  # 输入数据的特征维度
        self.output_size = 32  # 输出数据的维度

# 自注意力卷积 (Conv1D Attention)
class ConvAttention(layers.Layer):
    def __init__(self, filters, kernel_size):
        super(ConvAttention, self).__init__()
        self.conv1 = layers.Conv1D(filters, kernel_size, padding='same', activation='relu')
        # self.depthwise_conv1 = layers.DepthwiseConv1D(kernel_size, padding='same', activation='relu')
        # self.pointwise_conv1 = layers.Conv1D(filters, 1, activation='relu')  # 点卷积用于增加通道
        self.conv2 = layers.Conv1D(1, 1, activation='softmax')  # 注意力权重

    def call(self, inputs):
        # 提取局部特征
        local_features = self.conv1(inputs)  # [batch_size, time_steps, filters]
        # local_features = self.depthwise_conv1(inputs)
        # local_features = self.pointwise_conv1(local_features)
        # 计算注意力权重
        attention_weights = self.conv2(local_features)  # [batch_size, time_steps, 1]
        # 加权输入
        attention_output = inputs * attention_weights  # 按权重加权
        return attention_output

# 局部注意力机制 (Local Attention) 使用自注意卷积
class LocalAttention(layers.Layer):
    def __init__(self, filters, kernel_size=3):
        super(LocalAttention, self).__init__()
        self.conv_attention = ConvAttention(filters, kernel_size)

    def call(self, inputs):
        return self.conv_attention(inputs)

# 全局注意力机制 (Global Attention) 使用自注意卷积
class GlobalAttention(layers.Layer):
    def __init__(self, filters, kernel_size=3):
        super(GlobalAttention, self).__init__()
        self.conv_attention = ConvAttention(filters, kernel_size)
        self.global_pooling = layers.GlobalAveragePooling1D()

    def call(self, inputs):
        # 通过全局平均池化提取全局特征
        global_features = self.global_pooling(inputs)
        # 将全局特征广播到时间维度
        global_features_expanded = tf.expand_dims(global_features, axis=1)
        global_features_expanded = tf.tile(global_features_expanded, [1, tf.shape(inputs)[1], 1])  # [batch_size, time_steps, filters]
        # 对全局特征应用注意力
        global_attention_output = self.conv_attention(global_features_expanded)
        return global_attention_output

def SENet_Block(input_tensor, gamma=2, b=1):
    """
    input_tensor: 输入张量，形状为 (batch, time_steps, channels)
    gamma: 控制卷积核大小的参数，默认为 2
    b: 控制偏移量的参数，默认为 1

    返回值:
    输出张量，形状与输入相同
    """
    channels = input_tensor.shape[-1]
    # 自适应卷积核大小 k 的计算
    t = int(abs((tf.math.log(tf.cast(channels, tf.float32)) / tf.math.log(2.0)) + b) / gamma)
    k = t if t % 2 else t + 1  # 确保 k 为奇数
    # Global Average Pooling (通道上的全局平均池化)
    squeeze = tf.keras.layers.GlobalAveragePooling1D()(input_tensor)
    squeeze = tf.expand_dims(squeeze, axis=-1)  # 添加额外维度以适配 1D 卷积
    # 1D 卷积生成通道注意力权重
    excitation = tf.keras.layers.Conv1D(1, kernel_size=k, padding="same", use_bias=False)(squeeze)
    excitation = tf.keras.layers.Activation("sigmoid")(excitation)  # Sigmoid 激活
    excitation = tf.squeeze(excitation, axis=-1)  # 移除多余的维度
    # 按权重加权输入特征
    scale = tf.keras.layers.Multiply()([input_tensor, excitation])
    return scale

def compute_alpha(local_attention, global_attention):
    """
    计算融合权重 alpha，结合局部和全局特征共同参与权重计算。
    """
    combined_features = Concatenate(axis=-1)([local_attention, global_attention])
    fusion_weight = Dense(1, activation='sigmoid')(combined_features)  # 输出 alpha 的值 (0, 1)
    fused_features = Multiply()([fusion_weight, local_attention]) + Multiply()([1 - fusion_weight, global_attention])
    return fused_features

# 实例化args对象
args = Args()
bilstm = BiLSTM(args)
# Sequence Model 时序模型
def TDSFE(x_train, y_train, x_test, y_test, return_sequences=False):
    inputs = Input(shape=(128, 500))  # 输入形状：时间步长128，特征维度500
    # # 第一个 ResBlock + 局部注意力机制
    x = TSCN(inputs, filters=64, kernel_size=5, dilation_rate=1, F2=64)
    local_attention = LocalAttention(filters=64)(x)  # 局部注意力
    # # 第二个 ResBlock + 全局注意力机制
    x2 = TSCN(inputs, filters=32, kernel_size=3, dilation_rate=2, F2=32)
    global_attention = GlobalAttention(filters=32)(x2)  # 全局注意力
    # # # 计算融合权重 alpha，并进行融合
    fused_features = compute_alpha[local_attention, global_attention]

    x4 = bilstm(inputs)
    x4 = SENet_Block(x4)
    x4 = BatchNormalization()(x4)
    x4 = Dropout(0.2)(x4)

    x5 = Concatenate(axis=-1)([x4,fused_features])
    # 扁平化层与全连接层
    flatten_layer = Flatten()
    x5 = flatten_layer(x5)
    x5 = Dense(2, activation='softmax')(x5)  # 假设分类任务为二分类

    return Model(inputs=inputs, outputs=x5)



