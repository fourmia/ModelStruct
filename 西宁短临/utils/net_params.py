'''
Author: your name
Date: 2021-09-08 17:28:30
LastEditTime: 2021-09-08 17:47:09
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \A模型框架\西宁短临\utils\net_params.py
'''
import os
import sys
from collections import OrderedDict
#* 加载包路径
sys.path.append(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])
from 西宁短临.utils import configobj


def get_config0():
    dir_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
    config_path = '%s/config/para.config' % dir_path
    try:
        config = configobj.ConfigObj(config_path)
        config['Path']['lonlat']  = [int(ele) for ele in config['Path']['lonlat']]
        config['Path']['resolution'] = float(config['Path']['resolution'])
        config['Path']['threshold'] = float(config['Path']['threshold'])
    except:
        pass
    return config




# build model
conv2d_params = OrderedDict({
    'conv1_relu_1': [5, 64, 7, 5, 1],
    'conv2_relu_1': [64, 192, 5, 3, 1],
    'conv3_relu_1': [192, 192, 3, 2, 1],
    'deconv1_relu_1': [192, 192, 4, 2, 1],
    'deconv2_relu_1': [192, 64, 5, 3, 1],
    'deconv3_relu_1': [64, 64, 7, 5, 1],
    'conv3_relu_2': [64, 20, 3, 1, 1],
    'conv3_3': [20, 20, 1, 1, 0]
})


# build model
convlstm_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 8, 7, 5, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 192, 5, 3, 1]}),
        OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
    ],

    [
        ConvLSTM(input_channel=8, num_filter=64, b_h_w=(batch_size, 96, 96),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 32, 32),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 16, 16),
                 kernel_size=3, stride=1, padding=1),
    ]
]

convlstm_forecaster_params = [
    [
        OrderedDict({'deconv1_leaky_1': [192, 192, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [192, 64, 5, 3, 1]}),
        OrderedDict({
            'deconv3_leaky_1': [64, 8, 7, 5, 1],
            'conv3_leaky_2': [8, 8, 3, 1, 1],
            'conv3_3': [8, 1, 1, 1, 0]
        }),
    ],

    [
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 16, 16),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 32, 32),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=64, num_filter=64, b_h_w=(batch_size, 96, 96),
                 kernel_size=3, stride=1, padding=1),
    ]
]
