'''
Author: LZD
Date: 2021-09-08 11:14:57
LastEditTime: 2021-09-08 15:11:10
LastEditors: Please set LastEditors
Description: ConvLSTM Net Structure
FilePath: \A模型框架\XNnowcasting\models\convlstm.py
'''
import torch
import torch.nn as nn
from logzero import logger
import torch.nn.functional as F
from torch.types import Device


DEVICE = "pass"

class ConvLSTM(nn.Module):
    def __init__(self, input_channel, hidden_dim, kernel_size, batch_size, h_w, bias, stride=1, padding=1):

        super.__init__()
        self._conv = nn.Conv2d(in_channels=input_channel + hidden_dim,
                               out_channels= hidden_dim*4,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)
        self._input_channel = input_channel
        self._hidden_dim = hidden_dim
        self._kernel_size = kernel_size
        self._batch_size = batch_size
        self._bias = bias
        self._h, self._w = h_w

        #? 此处存在疑惑，从功能来看是初始化，两个版本实现不一致，前一版本模型未进行此步操作
        self.Wci = nn.Parameter(torch.zeros(1, self._hidden_dim, self._state_height, self._state_width)).to(DEVICE)
        self.Wcf = nn.Parameter(torch.zeros(1, self._hidden_dim, self._state_height, self._state_width)).to(DEVICE)
        self.Wco = nn.Parameter(torch.zeros(1, self._hidden_dim, self._state_height, self._state_width)).to(DEVICE)



    def forward(self, inputs=None, states=None, seq_len=10):
        """

        Args:
            inputs ([type], optional): [description]. Defaults to None.
            states ([type], optional): [description]. Defaults to None.
            seq_len (int, optional): [description]. Defaults to 10.
        """
        if states is None:
           h, c = self._init_hidden()
        else:
            h, c = states

        outputs = []
        for index in range(seq_len):    #? seq_len为时间序列长度
            if inputs is None:
                #? 此处h.size(0) 是否相当于 self._batch_size,且为啥要将输入全初始化为0
                logger.debug(f"batch_size:{self._batch_size}, h.size(0):{h.size(0)}")
                x = torch.zeros((h.size(0), self._input_channel, self._h, self._w),dtype=torch.float).to(DEVICE)
            else:
                x = inputs[index,...]
            cat_x = torch.cat([x, h],axis=1)    #? 此处操作按每个时间序列拼接
            conv_x = self._conv(cat_x)

            i, f, tmp_c, o = torch.chunk(conv_x, 4, dim=1)

            i = torch.sigmoid(i+self.Wci*c)
            f = torch.sigmoid(f+self.Wcf*c)
            c = f*c + i*torch.tanh(tmp_c)
            o = torch.sigmoid(o+self.Wco*c)
            h = o*torch.tanh(c)
            outputs.append(h)
        return torch.stack(outputs), (h, c)


    def _init_hidden(self):
        h = torch.zeros((self._batch_size, self._hidden_dim, self._h, self._w),dtype=torch.float).to(DEVICE)
        c = torch.zeros((self._batch_size, self._hidden_dim, self._h, self._w),dtype=torch.float).to(DEVICE)
        return h,c


