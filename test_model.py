import time
from util import *
import numpy as np
import sys
from models.googlenet import BasicConv2d

model = load_model('googlenet')
features_sum = 0
linear_sum = 0
for layer in model.layers:
    if isinstance(layer, BasicConv2d):
        features_sum += sys.getsizeof(layer.conv.weight.storage())
    elif isinstance(layer, torch.nn.Linear):
        linear_sum += sys.getsizeof(layer.weight.storage())
linear_sum = sys.getsizeof(model.fc.weight.storage())

print(features_sum, linear_sum)
# 58842480 494534800 vgg16


# x = torch.randn(1, *model.input_shape)
#
# res = []
#
# for i in range(5):
#     start = time.time()
#     o = model.forward_feature(x)
#     tf = time.time() - start
#     start = time.time()
#     _  = model.forward_classifier(o)
#     tc = time.time() - start
#
#     print(tf, tc, tf / (tf + tc))
#     if tc != 0:
#         res.append(tf / (tf + tc))
#
# print(np.mean(res))
# 0.9948153714537007 vgg16
# 0.9989992648396072 googlenet


