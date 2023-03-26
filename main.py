# This is a sample Python script.
import queue
import time
import torch
from models.googlenet import GoogLeNet, BasicConv2d
import torch.nn.functional as F
from net_analysis import output_input

if __name__ == '__main__':
    model = GoogLeNet()
    shape = 1, *model.input_shape
    print(shape)
    test = torch.randn(shape)
    # t = torch.randn(test.shape)
    start = time.time()
    _ = model(test)
    cost = time.time() - start
    print(f'{cost} seconds to inference googlenet')

    # test = torch.randn((1,1,3,3))
    # print(test.shape)
    # test = F.pad(test, (1,1,1,1))
    # print(test.shape)
    # layer = torch.nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True)
    # print(layer)

    # layer = BasicConv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)).conv
    # kernel = layer.weight
    #
    # x = torch.randn((1, 3, 224, 224))
    # p1 = (3, 0, 3, 3)
    # p2 = (0, 0, 3, 3)
    # p3 = (0, 2, 3, 3)
    # x1 = F.pad(x[:, :, :, 0:76], pad=p1)
    # x2 = F.pad(x[:, :, :, 71:150], pad=p2)
    # x3 = F.pad(x[:, :, :, 145:224], pad=p3)
    # o1 = F.conv2d(x1, weight=kernel, stride=(2, 2), padding=0)
    # o2 = F.conv2d(x2, weight=kernel, stride=(2, 2), padding=0)
    # o3 = F.conv2d(x3, weight=kernel, stride=(2, 2), padding=0)
    # print(o1.shape, o2.shape, o3.shape)
    # os = [o1, o2, o3]
    # output = torch.cat(os, dim=-1)
    # print(output.shape)
    # output2 = layer(x)
    # print(torch.equal(output, output2))
    #
    # layer = torch.nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True)
    # kernel_size = layer.kernel_size
    # x = output
    # padding = 0
    # x1 = x[:, :, :, 0:37]
    # x2 = x[:, :, :, 36:73]
    # x3 = x[:, :, :, 72:112]
    # o1 = F.max_pool2d(x1, kernel_size=kernel_size, stride=(2, 2), ceil_mode=True)
    # o2 = F.max_pool2d(x2, kernel_size=kernel_size, stride=(2, 2), ceil_mode=True)
    # o3 = F.max_pool2d(x3, kernel_size=kernel_size, stride=(2, 2), ceil_mode=True)
    # print(o1.shape, o2.shape, o3.shape)
    # os = [o1, o2, o3]
    # output = torch.cat(os, dim=-1)
    # print(output.shape)
    # output2 = layer(x)
    # print(torch.equal(output, output2))

    # a = [(1, [])]
    # def modify(b):
    #     b[0][1].append(1)
    # modify(a)
    # print(a)
