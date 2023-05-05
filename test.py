import pickle
import random
import sys
from queue import SimpleQueue

import torch

# test channel partition

import torch.nn.functional as F
from torch import nn
import torch
from paint import show_transmission_size
bod3 = [0, 0, 0, 0, 1204224, 200704, 0, 0, 0, 100352, 0, 100352, 401408, 0, 0, 0, 301056, 0, 200704, 752640, 150528, 0, 0, 0, 37632, 0, 50176, 125440, 0, 0, 0, 50176, 0, 50176, 100352, 0, 0, 0, 50176, 0, 50176, 87808, 0, 0, 0, 50176, 0, 50176, 200704, 0, 0, 0, 100352, 0, 100352, 326144, 50176, 0, 0, 0, 25088, 0, 25088, 75264, 0, 0, 0, 25088, 0, 25088, 1605632, 0, 802816, 802816, 802816, 827904, 0, 326144, 0]
mpbd3 = [28672, 0, 28672, 43008, 602112, 200704, 21504, 0, 0, 100352, 0, 100352, 401408, 28672, 0, 0, 301056, 0, 200704, 376320, 150528, 10752, 0, 0, 37632, 0, 50176, 125440, 12544, 0, 0, 50176, 0, 50176, 100352, 14336, 0, 0, 50176, 0, 50176, 87808, 16128, 0, 0, 50176, 0, 50176, 200704, 17920, 0, 0, 100352, 0, 100352, 163072, 50176, 8960, 0, 0, 25088, 0, 25088, 75264, 10752, 0, 0, 25088, 0, 25088, 802816, 53760, 401408, 401408, 401408, 413952, 46592, 163072, 0]
pcc3 = [143360, 0, 57344, 129024, 86016, 136192, 43008, 272384, 7168, 68096, 0, 68096, 272384, 57344, 408576, 14336, 204288, 0, 136192, 107520, 96768, 21504, 104832, 3584, 24192, 0, 32256, 80640, 25088, 112896, 5376, 32256, 0, 32256, 64512, 28672, 129024, 5376, 32256, 0, 32256, 56448, 32256, 145152, 7168, 32256, 0, 32256, 129024, 35840, 161280, 7168, 64512, 0, 64512, 93184, 35840, 17920, 44800, 3584, 17920, 0, 17920, 53760, 21504, 53760, 5376, 17920, 0, 17920, 630784, 1021440, 344064, 344064, 344064, 354816, 465920, 186368, 0]
ppc3 = [143360, 0, 57344, 129024, 86016, 0, 43008, 0, 7168, 0, 0, 0, 0, 57344, 0, 14336, 0, 0, 0, 107520, 0, 21504, 0, 3584, 0, 0, 0, 0, 25088, 0, 5376, 0, 0, 0, 0, 28672, 0, 5376, 0, 0, 0, 0, 32256, 0, 7168, 0, 0, 0, 0, 35840, 0, 7168, 0, 0, 0, 93184, 0, 17920, 0, 3584, 0, 0, 0, 0, 21504, 0, 5376, 0, 0, 0, 114688, 268800, 114688, 114688, 114688, 118272, 139776, 93184, 0]
assert len(bod3) == len(mpbd3) == len(pcc3) == len(ppc3)
xs = list(range(len(ppc3)))
show_transmission_size([bod3, mpbd3, pcc3, ppc3], ['BOD', 'MPBD', 'PCC', 'PPC'], 'transmission_size')
# a = 3551201
# b = 0.0654153 + 0.05106966 + 0.02884054
# print(b)
# a = 0.9186276
# b = 0.03401645 + 0.01035833 + 0.01231384
# print(b)
# x = torch.randn((1,2,3,3))
# layer = nn.Conv2d(2, 2, 3, 1, 1)
# weight = layer.weight
# print(weight.shape)
# out = layer(x)
# print(out.shape)
# weight1 = weight[:1]
# weight2 = weight[1:]
# out1 = F.conv2d(x, weight1, stride=1, padding=1)
# out2 = F.conv2d(x, weight2, stride=1, padding=1)
# out_ = torch.concat([out1, out2], dim=1)
# print(out)
# print(out_)
# print(out_.shape)
# print(torch.allclose(out, out_))
# from models.googlenet import BasicConv2d

# def product(num_list):
#     res = 1
#     for i in num_list:
#         res *= i
#     return res
#
#
# shape = [1, 64, 300, 150]
# a = torch.randn(shape)
#
# print(4 * product(shape))
# print(sys.getsizeof(a.storage()))
# print(len(pickle.dumps(a)) / 1024)
# b = a[..., -1:].clone().detach()
# print(b.shape)
# print(sys.getsizeof(b.storage()))
# print(len(pickle.dumps(b)) / 1024)


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

# q = SimpleQueue()
# print(q.get())


# import asyncio
# import aiohttp
#
# host = 'https://www.baidu.com'
# # urls_todo = {'/', '/1', '/2', '/3', '/4', '/5', '/6', '/7', '/8', '/9'}
#
# loop = asyncio.get_event_loop()
#
#
# async def fetch(url):
#     async with aiohttp.ClientSession(loop=loop) as session:
#         async with session.get(url) as response:
#             response = await response.read()
#             return response
#
#
# if __name__ == '__main__':
#     import time
#     start = time.time()
#     tasks = [fetch(host) for _ in range(10)]
#     res = loop.run_until_complete(asyncio.gather(*tasks))
#     print(time.time() - start)
#     print(res)



