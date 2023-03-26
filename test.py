from queue import SimpleQueue

import torch
import torch.nn.functional as F
from torch import nn
from models.googlenet import BasicConv2d

# layer = BasicConv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# last_output = torch.randn((1, 16, 28, 28))
# weight = layer.conv.weight
# partition = [0, 14, 28]
# paddings = [(1,0,1,1), (0,1,1,1)]
# stride = (1,1)
# require = [(0, 15), (13, 28)]
# worker = 0, 1
#
# correct_conv_output = layer.conv(last_output)
# correct_conv_output1 = correct_conv_output[..., 0:14]
# correct_output = layer(last_output)
# correct_output1 = correct_output[..., 0:14]
#
#
# x1 = last_output[..., 0:15]
# x1 = F.pad(x1, paddings[1])  # padding
# output1 = F.conv2d(x1, weight, stride=stride, padding=0)
# print(torch.equal(output1,correct_conv_output1))
# output1 = F.relu(output1, inplace=True)
# print(torch.equal(output1, correct_output1))

q = SimpleQueue()
print(q.get())
