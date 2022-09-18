from collections import OrderedDict


import torch
import torch.nn as nn

from utils_ import conv_kX2conv_kY,Identity2conv_kY,conv_kX_bn2conv_kX

#############################fuse 3*3 conv + Bn  to 3*3 conv ####################
print("#############################3*3 conv + Bn##################")
input_channel=3
output_channel=input_channel
kernel_size=3
padding_size=5

img_size=7
img=torch.randn(1,input_channel,img_size,img_size)

module = nn.Sequential(OrderedDict(
    conv=nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=kernel_size, stride=1, padding=padding_size, bias=False),  ##***默认没有偏差
    bn=nn.BatchNorm2d(num_features=input_channel)
))

module.eval() ##no update parameter
with torch.no_grad():
    result1=module(img)
    print("\nresult1:\n {}".format(result1))
    print("result1.shape :{} ".format(result1.shape))


W1=module.conv.weight

# print("bias:\n {}".format(module.conv.bias))
alphe1=module.bn.weight
beta1=module.bn.bias

var=module.bn.running_var
mean=module.bn.running_mean
eps=module.bn.eps
"""
t=alphe/sqrt(var+eps)
W_new=W1*t
bias_new=beta-mean*t
"""
std=(var+eps).sqrt()
t=alphe1/std

W_new=W1*t.reshape(-1, 1, 1, 1)
bias_new=beta1-mean*t

# print(W_new)
# print(bias_new)

conv2=nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=kernel_size, stride=1, padding=padding_size, bias=True)
conv2.load_state_dict(OrderedDict(weight=W_new, bias=bias_new))
print("input_channel:{}".format(conv2.in_channels))
with torch.no_grad():
    result2=conv2(img)
    print("\nresult2:\n {}".format(result2))
    print("result2.shape :{} ".format(result2.shape))




#################################conv_kX_bn2conv_kX   _function###################################
print("#############################conv_kX_bn2conv_kX   _function##################")


conv3=conv_kX_bn2conv_kX(module.conv,module.bn)
with torch.no_grad():
    result3=conv3(img)
    print("\nresult3:\n {}".format(result3))
    print("result3.shape :{} ".format(result3.shape))


##################conv_kX2conv_kY##################
print("\n\n##############conv_kX2conv_kY##############\n\n")

conv4=conv_kX2conv_kY(conv3,11)
with torch.no_grad():
    result4=conv4(img)
    print("\nresult4:\n {}".format(result4))
    print("result4.shape :{} ".format(result4.shape))

# print(result4==result1)



##Identity block  transform to conv_kY
print("\n\n################Identity block  transform to conv_kY##############\n\n")
conv5=Identity2conv_kY(3,5)

print(img)
print(conv5(img))