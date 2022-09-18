from collections import OrderedDict

import torch
from torch import nn
import numpy as np
def conv_kX_bn2conv_kX(conv2D_1,bn_1):

    padding_size=conv2D_1.padding
    input_channel=conv2D_1.in_channels
    kernel_size=conv2D_1.kernel_size

    W1 = conv2D_1.weight
    bias1=0
    if conv2D_1.bias is None:
        bias1=0

    alphe1 = bn_1.weight
    beta1 = bn_1.bias

    var =bn_1.running_var
    mean = bn_1.running_mean
    eps = bn_1.eps
    t=alphe1/((var+eps).sqrt())
    W_new=W1*t.reshape(-1,1,1,1)
    bias_new=beta1-t*mean+bias1

    conv_new=nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=kernel_size, stride=1, padding=padding_size, bias=True)
    conv_new.load_state_dict(OrderedDict(weight=W_new, bias=bias_new))
    return conv_new



def conv_kX2conv_kY(conv2d_1, result_y):
    """
    conv2d_1.weight .shape ==(output_channels, input_channels,X,X)  kernel_size ==X ,padding==pad_old

    result:
    conv2d_2.weight .shape ==(output_channels, input_channels,Y,Y)  kernel_size ==X ,padding==pad_old+(Y-X)/2

    e.g. 1*1 conv -->3*3 conv


    """

    input_channel = conv2d_1.in_channels
    pad_old=conv2d_1.padding[0]  ##tuple [0]
    k1 = conv2d_1.kernel_size[0]  ##tuple [0]
    k2 = result_y

    bias1 = conv2d_1.bias
    bias_flag=True
    if bias1 is None: bias_flag=False   ## bias == None will error

    weight1 = conv2d_1.weight

    pad_x = (int)((k2 - k1) / 2)

    f1 = weight1.detach().numpy()
    f1 = np.pad(f1, ((0, 0), (0, 0), (pad_x, pad_x), (pad_x, pad_x)), "constant")
    weight_new = torch.from_numpy(f1)

    conv2d_new = nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=k2, stride=1,
                           padding=pad_x+pad_old, bias=bias_flag)
    if bias_flag:
        conv2d_new.load_state_dict(OrderedDict(weight=weight_new, bias=bias1))
    else :
        conv2d_new.load_state_dict(OrderedDict(weight=weight_new))
    return conv2d_new



def Identity2conv_kY(input_channel,result_kernel_size):
    output_channel=input_channel
    f1=torch.zeros((output_channel,input_channel,1,1))
    for i in range(input_channel):
        for j in range(output_channel):
            if i == j:
                f1[i,j,...]=1
    # print(f1)
    conv2d_1=nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=1, stride=1, padding=0,bias=False)
    conv2d_1.load_state_dict(OrderedDict(weight=f1))

    conv2d_new=conv_kX2conv_kY(conv2d_1,result_kernel_size)
    return conv2d_new
