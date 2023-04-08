import pickle
import sys
from queue import SimpleQueue

import torch
# import torch.nn.functional as F
# from torch import nn
# from models.googlenet import BasicConv2d

def product(num_list):
    res = 1
    for i in num_list:
        res *= i
    return res


shape = [1, 3, 224, 224]
print(len(pickle.dumps(shape)))
a = torch.randn(shape)
b = torch.Tensor(a.storage())
print(b.shape)
c = sys.getsizeof(a.storage())
d = 4*product(shape)
e = len(pickle.dumps(a.storage()))
print(e - d)

print(a.storage().pickle_storage_type())

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
