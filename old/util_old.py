import math
from socket import *
import struct
import pickle
# import sklearn
import joblib
import numpy as np
import torch
import torch.nn as nn
from models.myNet import myNet
from models.LeNet5 import LeNet5
from models.vgg import vgg16, vgg11
from models.AlexNet import AlexNet, AlexNetbig
from joblib import load


def send_from(arr, dest: socket):
    ptr = memoryview(arr).cast('B')
    while len(ptr):
        nsent = dest.send(ptr)
        ptr = ptr[nsent:]


def recv_into(arr, source: socket):
    ptr = memoryview(arr).cast('B')
    while len(ptr):
        nrecv = source.recv_into(ptr)
        ptr = ptr[nrecv:]


def send_numpy(data, conn: socket, meta):
    src = memoryview(data).cast('B')
    data_len = len(src)
    print(f'Length of the data in bytes: {data_len}')
    header = struct.pack(_format, data_len, *meta)
    conn.sendall(header)
    while len(src):
        nsent = conn.send(src)
        src = src[nsent:]


def recv_numpy(conn: socket):
    header = conn.recv(_headsize)
    header = struct.unpack(_format, header)
    data_len = header[0]
    meta = header[1:]
    container = np.zeros(meta[2:], dtype=np.float32)
    dest = memoryview(container).cast('B')
    while len(dest):
        nrecv = conn.recv_into(dest)
        dest = dest[nrecv:]
    return meta, container


def list_layers(model: nn.Module):
    types = []
    for l in model.modules():
        if not isinstance(l, nn.Sequential):
            types.append(l.__class__)
    return types


# def execute_cmd(cmd: str) -> str:
#     return os.system(str)


# header: 1 int, 6 short
# fixed length header
_format = 'IHH'  # data_len, start_idx, end_idx
_headsize = struct.calcsize(_format)


def _send(data, conn: socket, meta):
    serialized = pickle.dumps(data)
    data_len = len(serialized)
    print(f'Send data size in bytes: {data_len}')
    header = struct.pack(_format, data_len, *meta)
    # print('header', header)
    ret = conn.sendall(header + serialized)
    if ret is not None:
        raise Exception("Some thing wrong with socket send!")


def _recv(conn: socket):
    header = conn.recv(_headsize)
    header = struct.unpack(_format, header)
    print(header)
    data_len = header[0]
    print(f'Recv data size in bytes: {data_len}')
    meta = header[1:]
    data_bytes = b''

    while data_len:
        recv = conn.recv(min(4096, data_len))
        data_bytes += recv
        data_len -= len(recv)
    data = pickle.loads(data_bytes)
    return meta, data


def load_model(model_name: str, dict_path=None, device='cpu'):
    model_name = model_name.lower()
    if model_name == 'mynet':
        model = myNet()

    elif model_name == 'lenet5':
        model = LeNet5()

    elif model_name == 'vgg16':
        model = vgg16()

    elif model_name == 'vgg11':
        model = vgg11()

    elif model_name == 'alexnet':
        model = AlexNet()

    elif model_name == 'alexnetbig':
        model = AlexNetbig()

    else:
        raise Exception('Invalid model name!')
    model.to(device)
    if dict_path is not None:
        model.load_state_dict(torch.load(dict_path, map_location=device))
    return model


def load_regressors():
    def zero(args):
        return 0

    return {'conv_pc': load('regress/models/conv_pc_regressor.pkl'),
            'conv_pi': load('regress/models/conv_pi_regressor_new.pkl'),
            'fc_pc': load('regress/models/fc_pc_regressor.pkl'),
            'fc_pi': load('regress/models/fc_pi_regressor.pkl'),
            'pool_pc': load('regress/models/pool_pc_regressor.pkl'),
            'pool_pi': load('regress/models/pool_pi_regressor.pkl'),
            'relu_pc': 0,
            'relu_pi': load('regress/models/relu_pi_regressor.pkl'),
            'drop_pc': 0,
            'drop_pi': load('regress/models/dropout_pi_regressor.pkl')
            }


def get_estimation(regressors, layer: str, device: str, args: list):
    label = f'{layer}_{device}'
    regressor = regressors[label]
    if regressor == 0:
        return 0
    return regressor.predict(np.array(args).reshape(1, -1))[0]


def latencies_estimation(model: nn.modules, input_shape: tuple, regressors: dict):
    shape = input_shape
    estimations_pi = []
    estimations_pc = []
    output_sizes = [np.product(input_shape) << 2]
    for i in range(model.depth):
        layer = model.get_layer(i)
        # shapes = []
        if isinstance(layer, nn.Conv2d):  # conv2d, linear, maxpool2d, relu, drop
            _in, _out, _kernel, _stride, _padding = layer.in_channels, layer.out_channels, layer.kernel_size, \
                layer.stride, layer.padding
            e = math.floor((shape[2] - _kernel[0] + 2 * int(_padding[0])) / _stride[0]) + 1
            # e2 = (shape[2] - _kernel[1] + 2 * int(_padding[1])) / _stride[1] + 1
            pixel = int((_kernel[0] / _stride[0]) ** 2 * _out)
            estimations_pi.append(get_estimation(regressors, 'conv', 'pi',
                                                 [_in*pixel, pixel]) * (shape[2] - _kernel[0]) * (shape[3] - _kernel[0]) / ((100 - _kernel[0]) ** 2))
            estimations_pc.append(get_estimation(regressors, 'conv', 'pc',
                                                 [_in*pixel, pixel]) * (shape[2] - _kernel[0]) * (shape[3] - _kernel[0]) / ((100 - _kernel[0]) ** 2))
            shape = (shape[0], _out, e, e)
        elif isinstance(layer, nn.Linear):
            estimations_pi.append(get_estimation(regressors, 'fc', 'pi', [layer.in_features, layer.out_features]))
            estimations_pc.append(get_estimation(regressors, 'fc', 'pc', [layer.in_features, layer.out_features]))
            shape = (shape[0], layer.out_features)
        elif isinstance(layer, nn.MaxPool2d):
            _kernel, _stride, _padding = layer.kernel_size, layer.stride, layer.padding
            if isinstance(_kernel, tuple):
                _kernel = _kernel[0]
            if isinstance(_stride, tuple):
                _stride = _stride[0]
            # print(_kernel, _stride, _padding)
            e = math.floor((shape[2] - _kernel + 2 * _padding) / _stride) + 1
            shape = (shape[0], shape[1], e, e)
            estimations_pi.append(get_estimation(regressors, 'pool', 'pi',
                                                 [np.product(shape), np.product(shape[0:2] * (e ** 2))]))
            estimations_pc.append(get_estimation(regressors, 'pool', 'pi',
                                                 [np.product(shape), np.product(shape[0:2] * (e ** 2))]))
        elif isinstance(layer, nn.ReLU):
            estimations_pi.append(get_estimation(regressors, 'relu', 'pi',
                                                 [np.product(shape)]))
            estimations_pc.append(get_estimation(regressors, 'relu', 'pc',
                                                 [np.product(shape)]))
        elif isinstance(layer, nn.Dropout):
            estimations_pi.append(get_estimation(regressors, 'drop', 'pi',
                                                 [np.product(shape)]))
            estimations_pc.append(get_estimation(regressors, 'drop', 'pc',
                                                 [np.product(shape)]))
        else:  # unsolved layers
            estimations_pi.append(0)
            estimations_pc.append(0)
        output_sizes.append(np.product(shape) << 2)
    return estimations_pi, estimations_pc, output_sizes


# def load_VGG16_latencies():
#     print('Loading VGG16 latency info...')
#     vgg_info = {}
#     with open('./model_device_latency', 'rb') as f:
#         vgg_info['model_pi_latency'] = pickle.load(f)
#     with open('./model_device_latency', 'rb') as f:
#         vgg_info['model_pc_latency'] = pickle.load(f)
#     with open('/model_sizes/vgg16_sizes.pkl', 'rb') as f:
#         vgg_info['output_size'] = pickle.load(f)
#     return vgg_info


# def execution_latency(device: torch.device, input, model):
#     if device.__str__() == 'cpu':
#         consumption = idx.idx()
#         _ = model(input)
#         consumption = (idx - consumption) * 100 # unit:ms
#         return consumption
#     elif device.__str__() == 'cuda':


# def send_model(conn: socket, partial_model: nn.Sequential, meta):
#     serialized = pickle.dumps(partial_model)
#     data_len = len(serialized)
#     header = struct.pack(_format, data_len, *meta)
#     ret = conn.sendall(header + serialized)
#     if ret is not None:
#         raise Exception("Something wrong with socket send!")
#
# def recv_model(conn: socket):
#     header = conn.recv(_headsize)
#     header = struct.unpack(_format, header)
