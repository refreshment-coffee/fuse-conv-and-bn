"""
just compare  the  time  of repvgg_block while training and inferencing

branch1： 3*3 conv+bn
branch2： 1*1 conv+bn
branch3： identity+bn

without Down sampling  conv block


so the result is not correct

repeat 1000 times :
train_time :  1.0874881744384766
inference_time :  0.07568001747131348

Theoretically, the inference time is half of the train time
"""

from collections import OrderedDict
import torch
import torch.nn as nn

from utils_ import conv_kX2conv_kY,Identity2conv_kY,conv_kX_bn2conv_kX

############################# ####################

input_channel=3
output_channel=input_channel
kernel_size=3
padding_size=1

img_size=7
img=torch.randn(1,input_channel,img_size,img_size)


import time
start_time = time.time()
train_time = 0
inference_time = 0
for i in range(1000):###repeat 1000 times ;;  you can set the time is to check the result is same or different .
    start_time = time.time()

    branch1=nn.Sequential(OrderedDict(
        conv=nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=3, stride=1,
                       padding=1, bias=False),
        bn=nn.BatchNorm2d(num_features=input_channel)
    ))

    branch2=nn.Sequential(OrderedDict(
        conv=nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=1, stride=1,
                       padding=0, bias=False),
        bn=nn.BatchNorm2d(num_features=input_channel)
    ))

    ##identity+bn
    branch3=nn.Sequential(OrderedDict(
        bn=nn.BatchNorm2d(num_features=input_channel)
    ))

    branch1.eval(),branch2.eval(),branch3.eval()
    with torch.no_grad():
        result1=branch1(img)
        result2=branch2(img)
        result3=branch3(img)

    result=result1+result2+result3
    train_time+=(time.time()-start_time)



    # print("\n result: ", result)          ##you can set the time is to check the result is same or different
    # print("\n result.shape: ", result.shape)

    #####before inference, the following operations only need to be done once !! so can ignore the following times
    conv1=conv_kX_bn2conv_kX(branch1.conv,branch1.bn)

    conv2=conv_kX_bn2conv_kX(conv_kX2conv_kY(branch2.conv,3),branch2.bn)

    conv3=conv_kX_bn2conv_kX(Identity2conv_kY(input_channel,3),branch3.bn)

    weight_new=conv1.weight+conv2.weight+conv3.weight
    bias_new=conv1.bias+conv2.bias+conv3.bias

    conv_new=nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=3, stride=1,
                       padding=1, bias=True)
    conv_new.load_state_dict(OrderedDict(weight=weight_new, bias=bias_new))
    #####before inference, the above operations only need to be done once !! so can ignore the following times

    start_time = time.time()

    with torch.no_grad():
        result4=conv_new(img)
    # print("\n result4: ", result4)         ## you can set the time is to check the result is same or different
    # print("\n result4.shape: ", result4.shape)
    inference_time+=(time.time()-start_time)


print("train_time : ",train_time)
print("inference_time : ", inference_time)