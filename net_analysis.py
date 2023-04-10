import queue
import sys
import threading
import time
import asyncio
import numpy as np
from torch import nn
from models.googlenet import BasicConv2d, GoogLeNet
from ExecutionUnit import ExecutionUnit
from util import *
from queue import SimpleQueue

# take GoogLeNet as DAG example
model = GoogLeNet()
layers = [
    model.conv1,  # 0
    model.maxpool1,  # 1
    model.conv2,  # 2
    model.conv3,  # 3
    model.maxpool2,  # 4
    model.inception3a.branch1,  # 5
    *model.inception3a.branch2,  # 67
    *model.inception3a.branch3,  # 89
    *model.inception3a.branch4,  # 1011
    # concat 70
    model.inception3b.branch1,  # 12
    *model.inception3b.branch2,  # 1314
    *model.inception3b.branch3,  # 1516
    *model.inception3b.branch4,  # 1718

    model.maxpool3,  # 19
    model.inception4a.branch1,  # 20
    *model.inception4a.branch2,  # 2122
    *model.inception4a.branch3,  # 2324
    *model.inception4a.branch4,  # 2526

    model.inception4b.branch1,  # 27
    *model.inception4b.branch2,  # 2829
    *model.inception4b.branch3,  # 3031
    *model.inception4b.branch4,  # 3233

    model.inception4c.branch1,  # 34
    *model.inception4c.branch2,  # 3536
    *model.inception4c.branch3,  # 3738
    *model.inception4c.branch4,  # 3940

    model.inception4d.branch1,  # 41
    *model.inception4d.branch2,  # 4243
    *model.inception4d.branch3,  # 4445
    *model.inception4d.branch4,  # 4647

    model.inception4e.branch1,  # 48
    *model.inception4e.branch2,  # 4950
    *model.inception4e.branch3,  # 5152
    *model.inception4e.branch4,  # 5354

    model.maxpool4,  # 55
    model.inception5a.branch1,  # 56
    *model.inception5a.branch2,  # 5758
    *model.inception5a.branch3,  # 5960
    *model.inception5a.branch4,  # 6162

    model.inception5b.branch1,  # 63
    *model.inception5b.branch2,  # 6465
    *model.inception5b.branch3,  # 6667
    *model.inception5b.branch4,  # 6869

    # model.avgpool,  # 70  avgpool with output_shape (1, 1) means kernel_size = input_shape
    # model.dropout,  # 71
    # model.fc,  # 72
    'concat',  # 73
    'concat',  # 74
    'concat',  # 75
    'concat',  # 76
    'concat',  # 77
    'concat',  # 78
    'concat',  # 79
    'concat',  # 80
    'concat',  # 81
]
# DAG of GoogLeNet
# next = [1, 2, 3, 4, [5, 6, 8, 10], 73, 7, 73, 9, 73, 11, 73, 74, 14, 74, 16, 74, 18, 74, [20, 21, 23, 25], 75, 22, 75,
#         24, 75, 26, 75, 76, 29, 76, 31, 76, 33,
#         76, 77, 36, 77, 38, 77, 40, 77, 78, 43, 78, 45, 78, 47, 78, 79, 50, 79, 52, 79, 54, 79, [56, 57, 59, 61], 80,
#         58, 80, 60, 80, 62, 80, 81, 65, 81,
#         67, 81, 69, 81, 71, 72, [], [12, 13, 15, 17], 19, [27, 28, 30, 32], [34, 35, 37, 39], [41, 42, 44, 46],  # whole network
#         [48, 49, 51, 53], 55, [63, 64, 66, 68], 70]
next = [1, 2, 3, 4, [5, 6, 8, 10], 70, 7, 70, 9, 70, 11, 70, 71, 14, 71, 16, 71, 18, 71, [20, 21, 23, 25], 72, 22, 72,
        24, 72, 26, 72, 73, 29, 73, 31, 73, 33,
        73, 74, 36, 74, 38, 74, 40, 74, 75, 43, 75, 45, 75, 47, 75, 76, 50, 76, 52, 76, 54, 76, [56, 57, 59, 61], 77,
        58, 77, 60, 77, 62, 77, 78, 65, 78,
        67, 78, 69, 78, [12, 13, 15, 17], 19, [27, 28, 30, 32], [34, 35, 37, 39], [41, 42, 44, 46],
        # remove the last several layers like adaptive average pool and fully connected layers
        [48, 49, 51, 53], 55, [63, 64, 66, 68], []]

for i in range(len(next)):  # 将next数组内的单个元素处理为长度为1的列表
    if not isinstance(next[i], list):
        next[i] = [next[i]]


# print(len(next))


def next_to_last(next):  # 将next数组转化为last数组，即last数组
    total = len(next)
    last = [[] for _ in range(total)]
    # last[0].append(-1)  # -1 represents the original input
    for i, nexts in enumerate(next):
        for l in nexts:
            last[l].append(i)
    return last


# print(len(layers))
# layers_dependency = next_to_last(next)


def topology_DAG(next_array, last_array):  # transfer the DAG network to topology list, starts from 0, bfs
    total = len(next_array)
    in_num = np.zeros(total)
    for i in range(total):
        in_num[i] = len(last_array[i])
    q = queue.Queue()
    q.put(0)
    ans = []
    while not q.empty():
        ele = q.get()
        ans.append(ele)
        if isinstance(next_array[ele], list):
            for i in next_array[ele]:
                in_num[i] -= 1
                if in_num[i] == 0:
                    q.put(i)
        else:
            in_num[next_array[ele]] -= 1
            if in_num[next_array[ele]] == 0:
                q.put(next_array[ele])
    return ans


# average distribution
# output features range of layer


def cal_output_shape(net, topology_list, last_array):
    n_layers = len(topology_list)
    output_shapes = [[] for _ in range(n_layers)]
    mark = np.zeros(n_layers)
    for lth in topology_list:
        mark[lth] = 1
        if layers[lth] == 'concat':
            inputs = []
            for last in last_array[lth]:
                inputs.append(torch.randn(output_shapes[last]))
            output = torch.cat(inputs, 1)

        else:
            if lth == 0:
                input_shape = 1, *net.input_shape
            else:
                input_shape = output_shapes[last_array[lth][0]]
            x = torch.randn(input_shape)
            if isinstance(layers[lth], nn.Linear):
                x = torch.flatten(x, 1)
            output = layers[lth](x)
        output_shapes[lth] = output.shape

    return output_shapes


def cal_output(topology_list, last_array, x):
    outputs = [None for _ in range(n_layers)]
    mark = np.zeros(n_layers)
    for lth in topology_list:
        if lth == 0:
            outputs[0] = layers[0](x)
            continue
        mark[lth] = 1
        if layers[lth] == 'concat':
            inputs = [outputs[i] for i in last_array[lth]]
            output = torch.cat(inputs, 1)
        else:
            assert len(last_array[lth]) == 1
            last_layer = last_array[lth][0]
            output = layers[lth](outputs[last_layer])
        outputs[lth] = output
    return outputs


# layers_output_shapes = cal_output_shape(model, topology_layers, layers_dependency)


# def cal_inputFromOutput(output_shapes, last_layers):
#     n_layers = len(output_shapes)
#     input_shapes = [[] for _ in range(n_layers)]
#     for nl in topology_layers:
#         if nl == 0:
#             input_shape = [1, 3, 224, 224]
#         else:
#             lasts = last_layers[nl]
#             if len(lasts) == 1:
#                 last = lasts[0]
#                 input_shape = output_shapes[last]
#             else:  # have over 1 last layers
#                 input_shape = []
#                 for last in lasts:
#                     input_shape.append(output_shapes[last])
#         input_shapes[nl] = input_shape
#     return input_shapes
#
#
# # compute the layers' output shape and store in model
# model.input_shapes = cal_inputFromOutput(layers_output_shapes, layers_dependency)


# print(model.output_shapes)


# partitioning dimension: -1 # 假设从最后一维开始切
def workload_split(num_device):  # temporarily average
    partitions = []
    for i, shape in enumerate(model.output_shapes):
        if layers[i] == 'concat':
            partitions.append(1)
        else:
            length = shape[-1]
            partition = [0 for _ in range(num_device + 1)]
            partition[-1] = length
            average = length // num_device
            for i in range(1, num_device):
                partition[i] = average * i
            partitions.append(partition)
    return partitions


# from output range to input range
def output_input(output_range: tuple, layer_config=None) -> tuple:
    o_s, o_e = output_range
    layer_type = layer_config['type']
    if layer_type == 'relu':  # most activation layers
        return output_range
    elif layer_type == 'upsample':
        scale_factor = layer_type['scale_factor']
        return round(o_s / scale_factor), round(o_e / scale_factor)
    elif layer_type in ('conv', 'basicConv', 'maxpool'):
        kernel_size, stride, padding = layer_config['kernel_size'], layer_config['stride'], layer_config['padding']
        if padding != 0:
            padding = padding[1]
        return o_s * stride[1] - padding, (o_e - 1) * stride[1] + kernel_size[1] - padding
    # elif layer_type == 'pooling':
    #     kernel_size, stride, padding, ceil_mode = layer_config['kernel_size'], layer_config['stride'], layer_config['padding'], layer_config['ceil_mode']
    #     if padding != 0:
    #         padding = padding[1]
    #     return
    else:
        print('Unknown layer type')


# generate execution units
def gen_inputDependency(layer_list, topology_list, output_partitions, last_array):
    # required input: from which layer, input range(in the -1 dimension )
    ids = [list() for _ in range(len(topology_list))]
    for l in topology_list:  # 当前这层
        partition = output_partitions[l]

        if layers[l] == 'concat':
            last_division = tuple(len(partitions[last_layer]) - 1 for last_layer in last_array[l])
            required_input = (last_array[l], last_division)
            layer_config = {'type': 'concat'}
            ids[l].append((required_input, layer_config, []))
        else:
            if l == 0:
                H = model.input_shape[-1]
            else:
                H = model.output_shapes[last_array[l][0]][-1]
            for i in range(len(partition) - 1):
                # get output range
                output_range = partition[i: i + 2]  # [o_s, o_e)

                # #tricky
                # if l == i == 0:
                #     output_range[1] = 151

                layer = layer_list[l]
                # get corresponding input range
                if isinstance(layer, (nn.Conv2d, BasicConv2d)):
                    type = 'conv'
                    if isinstance(layer, BasicConv2d):
                        layer = layer.conv
                        type = 'basicConv'
                    layer_config = {'type': type, 'kernel_size': layer.kernel_size, 'stride': layer.stride,
                                    'padding': layer.padding}
                    i_s, i_e = input_range = output_input(output_range, layer_config)  # [i_s, i_e)
                    if layer.padding == 0:
                        padding = (0, 0, 0, 0)
                    else:
                        if i_s < 0:
                            upper_padding = -i_s
                            i_s = 0
                        else:
                            upper_padding = 0
                        if i_e > H:
                            bottom_padding = i_e - H
                            i_e = H
                        else:
                            bottom_padding = 0
                        padding = (upper_padding, bottom_padding, *layer.padding)
                        input_range = (i_s, i_e)
                    layer_config['padding'] = padding

                elif isinstance(layer, nn.MaxPool2d):  # padding = 0 for most maxpool layers
                    layer_config = {'type': 'maxpool', 'kernel_size': layer.kernel_size, 'stride': layer.stride,
                                    'padding': layer.padding, 'ceil_mode': layer.ceil_mode}
                    i_s, i_e = input_range = output_input(output_range, layer_config)  # [i_s, i_e)

                    if layer.padding == 0:
                        padding = (0, 0)
                    else:
                        padding = layer.padding
                    # else:
                    if i_s < 0:
                        upper_padding = -i_s
                        i_s = 0
                    else:
                        upper_padding = 0
                    if i_e > H:
                        bottom_padding = i_e - H
                        i_e = H
                    else:
                        bottom_padding = 0
                    padding = (upper_padding, bottom_padding, *padding)
                    input_range = (i_s, i_e)
                    layer_config['padding'] = padding

                elif isinstance(layer, nn.Upsample):
                    layer_config = {'type': 'upsample', 'scale_factor': layer.scale_factor}
                    input_range = output_input(output_range, layer_config)
                elif isinstance(layer, (nn.Sigmoid, nn.ReLU, nn.Dropout, nn.BatchNorm2d)):
                    layer_config = {'type': 'bijective'}
                    input_range = output_range
                else:
                    input_range = None
                    layer_config = None

                required_input = (last_array[l], input_range)
                ids[l].append((required_input, layer_config, []))

    return ids


# for w in workload_dependency:
#     print(w)


# def gen_forwarding(input_dependency: list, topology_list: list, dependency_list: list):
#     for nl in topology_list:  # this layer
#         lasts = dependency_list[nl]
#         if len(lasts) == 1:  # only depends on one layer, can be the next layer of concat
#             for i, n in enumerate(input_dependency[nl]):  # execution units of this layers
#                 last_layer, input_range = n[0]
#                 last_layer = last_layer[0]
#                 last_partition = partitions[last_layer]  # partition of last layer's output
#                 if last_partition == 1:  # last layer is concat, has whole input
#                     input_dependency[last_layer][0][2].append((i, input_range))
#                 else:
#                     # formation = []  # the formation of layer's input
#                     for j in range(len(last_partition) - 1):
#                         left_max = max(last_partition[j], input_range[0])
#                         right_min = min(last_partition[j + 1], input_range[1])
#                         if left_max < right_min:  # overlap
#                             overlap = (left_max, right_min)
#                             input_dependency[last_layer][j][2].append((i, overlap))
#                             # formation.append((n, overlap))
#         else:  # concat layer
#             for last in lasts:
#                 last_partition = partitions[last]
#                 for i in range(len(last_partition) - 1):
#                     input_dependency[last][i][2].append((0, (partitions[last][i], partitions[last][i + 1])))


def gen_forwarding(n_device: int, input_dependency, topology_list, next_layers, output_partitions):
    for l in topology_list:  # layer nl
        if l == 2:
            h = 1
        nexts = next_layers[l]  # next layers
        partition = output_partitions[l]
        if len(nexts) == 0:  # output of final layer should send back to master
            continue
            # input_dependency[l][0][2].append(1)
        if layers[l] == 'concat':  # layer l is concat
            forwarding = [[] for _ in range(n_device)]
            for nl in nexts:
                for i, eu in enumerate(input_dependency[nl]):
                    input_range = eu[0][1]
                    forwarding[i].append(input_range)
            for to_device, f in enumerate(forwarding):  # d为设备号
                if len(f) > 0:  # 按照转发对应的设备号去重
                    for interval in get_set_union(f):
                        input_dependency[l][0][2].append((to_device, interval))
        elif len(nexts) == 1 and layers[nexts[0]] == 'concat':  # next layer is concat
            for i in range(len(partition) - 1):
                # input_dependency[l][i][2].append((0, 1))  # 表示把所有输出转给设备0
                input_dependency[l][i][2].append((0, (0, partition[i + 1] - partition[i]), partition[i]))
        else:  # no concat layer between this layer and next layer
            # forwarding = [[] for _ in range(n_device)]  # 未去重的forwarding
            for nl in nexts:
                for i in range(len(partition) - 1):
                    forwarding = [[] for _ in range(n_device)]
                    for j, eu in enumerate(input_dependency[nl]):
                        _, input_range = eu[0]
                        overlap = get_intersection((partition[i], partition[i + 1]), input_range)
                        if overlap is not None:
                            forwarding[j].append(overlap)
                            # input_dependency[nl][i][2].append((j, overlap))
                    for to_device, f in enumerate(forwarding):  # d为设备号
                        if len(f) > 0:  # 按照转发对应的设备号去重
                            for interval in get_set_union(f):
                                input_dependency[l][i][2].append(
                                    (to_device, (interval[0] - partition[i], interval[1] - partition[i]), partition[i]))


# for eu in workload_dependency:
#     print(eu)


def gen_executionUnits(n_device: int, workload_partition, topology_list):
    device_group = [[] for _ in range(n_device)]
    for l in topology_list:  # current layer
        for i, eu in enumerate(workload_partition[l]):
            device_group[i].append(
                ExecutionUnit(required_input=eu[0], operator=eu[1], forwarding=eu[2], layer_num=l, device_num=i))
    return device_group


def execute(tq: SimpleQueue, ri: list, worker_no: int, result_list: list):
    # global output
    print(f'this is worker {worker_no}')
    while not tq.empty():

        task = tq.get()
        layer_no = task.layer_num
        required_input = input_satisfactory(task.required_input, ri)
        start = time.time()
        while required_input is None:
            time.sleep(0.1)
            required_input = input_satisfactory(task.required_input, ri)
            if time.time() - start > 3:
                print(f'layer {layer_no} on device {worker_no} is blocked')
                return
        # print(f'input of layer {layer_no} device {worker_no} is ready')
        layer = layers[layer_no]
        if isinstance(layer, BasicConv2d):
            output = task.execute(required_input, layer.conv.weight)
        # elif isinstance(layer, torch.nn.Conv2d):
        #     output = task.execute(required_input, layer.weight)
        else:
            output = task.execute(required_input)

        partition = partitions[layer_no]
        if partition == 1:
            correct = outputs[layer_no]
        else:
            correct = outputs[layer_no][..., partition[worker_no]:partition[worker_no + 1]]
        same = torch.equal(output, correct)
        print(f'layer {layer_no} on device {worker_no} is {same}')

        if not same:
            h = 1

        for f in task.forwarding:
            if len(f) == 2:
                to_device, interval = f
                l, r = interval
            else:
                to_device, interval, left = f
                l, r = interval[0] + left, interval[1] + left
            # left = interval[-1]
            recv_input[to_device][layer_no].append(((l, r), output[..., interval[0]:interval[1]]))  # 对于该层输出的真实范围以及对应数据
        # print(f'task of layer {layer_no} device {worker_no} has finished')
    result_list.append((worker_no, output))


if __name__ == '__main__':
    model.input_shape = 3, 600, 600
    layers_dependency = next_to_last(next)
    n_layers = len(layers_dependency)
    topology_layers = topology_DAG(next, layers_dependency)

    # print(f'layers\' topology list: {topology_layers}')
    # print(f'layers\' dependency   : {layers_dependency}')

    n_device = 2
    model.output_shapes = cal_output_shape(model, topology_layers, layers_dependency)
    # for i in range(len(model.output_shapes)):
    # print(f'{i}: {model.output_shapes[i]}')

    partitions = workload_split(num_device=n_device)
    # partitions[1][1] = 74
    for i, partition in enumerate(partitions):
        print(f'{i}: {partition}')
    # print(partitions[73:])

    workload_dependency = gen_inputDependency(layers, topology_layers, partitions, layers_dependency)

    gen_forwarding(n_device, workload_dependency, topology_layers, next, partitions)

    # useless
    # workload_dependency[0] = [(([], (0, 304)),
    #                            {'type': 'basicConv', 'kernel_size': (7, 7), 'stride': (2, 2), 'padding': (3, 0, 3, 3)},
    #                            [(0, (0, 151), 0)]),
    #                           (([], (297, 600)),
    #                            {'type': 'basicConv', 'kernel_size': (7, 7), 'stride': (2, 2), 'padding': (0, 2, 3, 3)},
    #                            [(1, (0, 150), 150)])]
    #
    # workload_dependency[1] = [(([0], (0, 151)),
    #                            {'type': 'maxpool', 'kernel_size': (3, 3), 'stride': (2, 2), 'padding': (0, 0, 0, 0),
    #                             'ceil_mode': True}, [(0, (0, 75), 0)]),
    #                           (([0], (150, 300)),
    #                            {'type': 'maxpool', 'kernel_size': (3, 3), 'stride': (2, 2), 'padding': (0, 1, 0, 0),
    #                             'ceil_mode': True}, [(1, (0, 75), 75)])]

    for layer, i in enumerate(workload_dependency):
        print(f'layer {layer} {i}')

    # forwarding_sizes = [[] for _ in range(n_layers)]
    # for layer, i in enumerate(workload_dependency):
    #     for device, eu in enumerate(i):
    #         fs = []
    #         for f in eu[2]:
    #             if f[0] != device:
    #                 shape = (*model.output_shapes[layer][:-1], f[1][1] - f[1][0])
    #                 # print(shape)
    #                 fs.append(cal_tensor_size(shape))
    #         forwarding_sizes[layer].append(fs)
    #
    # print(forwarding_sizes)

    execution_units = gen_executionUnits(n_device, workload_dependency, topology_layers)
    # for eu in execution_units:
    #     print(len(eu))

    # 在本地模拟DNN拆分子任务执行判断拆分是否正确

    task_queue = [SimpleQueue() for _ in range(n_device)]  # store all tasks
    # execute_queue = [SimpleQueue() for _ in range(n_device)]  # store input-ready tasks
    recv_input = [[[] for _ in range(n_layers + 1)] for _ in range(n_device)]  # item: (input, from_layer)

    for device, nl in enumerate(execution_units):  # put tasks into corresponding device queue
        for eu in nl:
            task_queue[device].put(eu)

    required_inputs = [execution_units[i][0].required_input[1] for i in range(n_device)]
    print(required_inputs)
    x = torch.randn((1, *model.input_shape))  # test input
    outputs = cal_output(topology_layers, layers_dependency, x)
    # how to store the recv input and how to judge required input is satisfied
    simulate = True
    if not simulate:
        sys.exit(0)
    local = False
    if local:
        for device, ri in enumerate(required_inputs):
            recv_input[device][-1].append(x[..., ri[0]:ri[1]])  # cut from the last dimension of 4D input

        results = []
        threads = []
        for w in range(n_device):
            threads.append(threading.Thread(target=execute, args=[task_queue[w], recv_input[w], w, results]))
        for worker in threads:
            worker.start()
        for worker in threads:
            worker.join()
        results.sort(key=lambda item: item[0])
        result = results[0][1]
        last_output = outputs[-1]
        print(torch.allclose(result, last_output, rtol=0, atol=1e-16))
        # print(right_output.shape)
        print(result.shape)
        # print(torch.equal(right_output, result))
    else:  # online testing
        from master import Master

        m = Master(num_required_worker=n_device)
        m.start()
        time.sleep(2)
        first_inputs = [x[..., ri[0]:ri[1]].clone().detach() for ri in required_inputs]
        loop = asyncio.get_event_loop()
        print('Send subtasks and input to workers...')
        try:
            send_tasks = [async_send_data(m.worker_sockets[device], eus) for device, eus in enumerate(execution_units)]
            loop.run_until_complete(asyncio.gather(*send_tasks))
        except Exception as e:
            print(f'Error occurred when send subtasks:\n{e}')
        # for device, eus in enumerate(execution_units):
        #     try:
        #         send_data(m.worker_sockets[device], eus)
        #     except Exception as e:
        #         print(f'Error occurred when send subtasks:\n{e}')

        start = time.time()  # worker接受所有tasks需要花很多时间
        try:
            send_tasks = [async_send_data(m.worker_sockets[device], first_input) for device, first_input in
                          enumerate(first_inputs)]
            loop.run_until_complete(asyncio.gather(*send_tasks))
        except Exception as e:
            print(f'Error occurred when send required input to workers:\n{e}')
        # for device, first_input in enumerate(first_inputs):
        #     try:
        #         send_data(m.worker_sockets[device], first_input)
        #     except Exception as e:
        #         print(f'Error occurred when send subtasks: {e}')

        try:
            data = recv_data(m.worker_sockets[0])
            consumption = time.time() - start
            print(torch.allclose(outputs[-1], data))
            print(f'Recv final result in {consumption}s')
        except timeout:
            print('Recv results time out!')
        except Exception as e:
            print(e)
