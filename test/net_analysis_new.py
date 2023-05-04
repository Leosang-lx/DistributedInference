# total partition
import asyncio
import queue
import random
import sys
import threading
import time

from torch import nn

from ExecutionUnit import ExecutionUnit
from models.googlenet import BasicConv2d
from paint import *
from util import *

# take GoogLeNet as DAG example
model = load_model('googlenet')
layers = model.layers


def translate_next_array(next_array):
    for i in range(len(next_array)):  # 将next数组内的单个元素处理为长度为1的列表
        if not isinstance(next_array[i], list):
            next_array[i] = [next_array[i]]


def next_to_last(next):  # 将next数组转化为last数组，即last数组
    total = len(next)
    last = [[] for _ in range(total)]
    # last_array[0].append(-1)  # -1 represents the original input
    for i, nexts in enumerate(next):
        for l in nexts:
            last[l].append(i)
    return last


# print(len(layers))
# layers_dependency = next_to_last(next_array)


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
            this_layer = net.layers[lth]
            if isinstance(this_layer, nn.Linear):
                x = torch.flatten(x, 1)
            output = this_layer(x)
        output_shapes[lth] = output.shape

    return output_shapes


def cal_output(topology_list, last_array, x):
    n_layers = len(topology_list)
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
#                 last_array = lasts[0]
#                 input_shape = output_shapes[last_array]
#             else:  # have over 1 last_array layers
#                 input_shape = []
#                 for last_array in lasts:
#                     input_shape.append(output_shapes[last_array])
#         input_shapes[nl] = input_shape
#     return input_shapes
#
#
# # compute the layers' output shape and store in model
# model.input_shapes = cal_inputFromOutput(layers_output_shapes, layers_dependency)


# print(model.output_shapes)


# partitioning dimension: -1 # 假设从最后一维开始切
def workload_partition(output_shapes, num_device):  # temporarily average
    partitions = []
    for i, shape in enumerate(output_shapes):
        # if layers[i] == 'concat':
        #     partitions.append(1)
        # else:
        length = shape[-1]
        partition = [0 for _ in range(num_device + 1)]
        partition[-1] = length
        average = round(length / num_device)
        for i in range(1, num_device):
            partition[i] = average * i
        partitions.append(partition)
    return partitions


def random_workload_partition(output_shapes, num_device):
    partitions = []
    for i, shape in enumerate(output_shapes):
        length = shape[-1]
        partition = []
        cnt = 0
        while cnt < num_device - 1:
            p = random.randint(1, length - 1)
            if p not in partition:
                cnt += 1
                partition.append(p)
        partition.sort()
        partition = [0, *partition, length]
        partitions.append(partition)
    return partitions


# from output range to input range
def output_input(output_range: tuple, layer_config=None) -> tuple:
    o_s, o_e = output_range
    layer_type = layer_config['type']
    if layer_type in ['relu', 'concat']:  # most activation layers
        return output_range
    elif layer_type == 'upsample':
        scale_factor = layer_type['scale_factor']
        return round(o_s / scale_factor), round(o_e / scale_factor)
    elif layer_type in ('conv', 'basicConv', 'maxpool'):
        kernel_size, stride, padding = layer_config['kernel_size'], layer_config['stride'], layer_config['padding']
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if padding != 0:
            padding = padding[1]
        return o_s * stride[1] - padding, (o_e - 1) * stride[1] + kernel_size[1] - padding
    else:
        print('Unknown layer type')


# generate execution units
def gen_inputDependency(layer_list, topology_list, output_partitions, last_array):
    # required input: from which layer, input range(in the -1 dimension )
    ids = [list() for _ in range(len(topology_list))]
    for l in topology_list:  # 当前这层
        partition = output_partitions[l]

        # if layers[l] == 'concat':
        #     last_division = tuple(len(partitions[last_layer]) - 1 for last_layer in last_array[l])
        #     required_input = (last_array[l], last_division)
        #     layer_config = {'type': 'concat'}
        #     ids[l].append((required_input, layer_config, []))
        # else:
        if l == 0:
            H = model.input_shape[-1]
        else:
            H = model.output_shapes[last_array[l][0]][-1]
        for i in range(len(partition) - 1):
            # get output range
            output_range = partition[i: i + 2]  # [o_s, o_e)
            layer = layer_list[l]
            # get corresponding input range
            if isinstance(layer, BasicConv2d):
                # type = 'conv'
                # if isinstance(layer, BasicConv2d):
                conv = layer.conv
                # bn = layer.bn
                type = 'basicConv'
                layer_config = {'type': type, 'kernel_size': conv.kernel_size, 'stride': conv.stride,
                                'padding': conv.padding}  #, 'bn_args': (bn.weight, bn.bias, False, bn.momentum, bn.eps)}
                i_s, i_e = input_range = output_input(output_range, layer_config)  # [i_s, i_e)
                if conv.padding == 0:
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
                    padding = (upper_padding, bottom_padding, *conv.padding)
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
            elif layer == 'concat':
                layer_config = {'type': layer}
                input_range = output_range
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
#         if len(lasts) == 1:  # only depends on one layer, can be the next_array layer of concat
#             for i, n in enumerate(input_dependency[nl]):  # execution units of this layers
#                 last_layer, input_range = n[0]
#                 last_layer = last_layer[0]
#                 last_partition = partitions[last_layer]  # partition of last_array layer's output
#                 if last_partition == 1:  # last_array layer is concat, has whole input
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
#             for last_array in lasts:
#                 last_partition = partitions[last_array]
#                 for i in range(len(last_partition) - 1):
#                     input_dependency[last_array][i][2].append((0, (partitions[last_array][i], partitions[last_array][i + 1])))


def gen_forwarding(n_device: int, input_dependency, topology_list, next_layers, output_partitions):
    for l in topology_list:  # layer nl
        nexts = next_layers[l]  # next_array layers
        partition = output_partitions[l]
        if len(nexts) == 0:  # output of final layer should send back to master
            continue
            # input_dependency[l][0][2].append(1)
        # if layers[l] == 'concat':  # layer l is concat
        #     forwarding = [[] for _ in range(n_device)]
        #     for nl in nexts:
        #         for i, eu in enumerate(input_dependency[nl]):
        #             input_range = eu[0][1]
        #             forwarding[i].append(input_range)
        #     for to_device, f in enumerate(forwarding):  # d为设备号
        #         if len(f) > 0:  # 按照转发对应的设备号去重
        #             for interval in get_set_union(f):
        #                 input_dependency[l][0][2].append((to_device, interval))
        # elif len(nexts) == 1 and layers[nexts[0]] == 'concat':  # next_array layer is concat
        #     for i in range(len(partition) - 1):
        #         input_dependency[l][i][2].append((0, (0, partition[i + 1] - partition[i]), partition[i]))
        # else:  # no concat layer between this layer and next_array layer
        for nl in nexts:
            for i in range(len(partition) - 1):
                forwarding = [[] for _ in range(n_device)]  # 未去重的forwarding
                for j, eu in enumerate(input_dependency[nl]):
                    _, input_range = eu[0]
                    overlap = get_intersection((partition[i], partition[i + 1]), input_range)
                    if overlap is not None:
                        forwarding[j].append(overlap)
                for to_device, f in enumerate(forwarding):  # d: device_id
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
        if layer_no == 70:
            h = 1
        required_input = input_satisfactory3(task.required_input, ri)
        start = time.time()
        while required_input is None:
            time.sleep(0.1)
            required_input = input_satisfactory3(task.required_input, ri)
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
    model.input_shape = 3, 224, 224
    next = model.next
    translate_next_array(next)
    layers_dependency = next_to_last(next)
    n_layers = len(layers_dependency)
    topology_layers = topology_DAG(next, layers_dependency)

    # print(f'layers\' topology list: {topology_layers}')
    # print(f'layers\' dependency   : {layers_dependency}')

    n_device = 2
    model.output_shapes = cal_output_shape(model, topology_layers, layers_dependency)
    # for i in range(len(model.output_shapes)):
    # print(f'{i}: {model.output_shapes[i]}')

    partitions = workload_partition(model.output_shapes, n_device)
    # partitions[1][1] = 74
    for i, partition in enumerate(partitions):
        print(f'{i}: {partition}')
    # print(partitions[73:])

    workload_dependency = gen_inputDependency(layers, topology_layers, partitions, layers_dependency)

    gen_forwarding(n_device, workload_dependency, topology_layers, next, partitions)

    for layer, i in enumerate(workload_dependency):
        print(f'layer {layer} {i}')

    execution_units = gen_executionUnits(n_device, workload_dependency, topology_layers)
    # for eu in execution_units:
    #     print(len(eu))

    # cal the minimal transmission size of each worker
    forward_workers = [[] for _ in range(n_device)]  # unit: byte
    for w, eus in enumerate(execution_units):
        for eu in eus:
            assert isinstance(eu, ExecutionUnit)
            l = eu.layer_num
            f_size = 0
            for f in eu.forwarding:
                if f[0] != w:
                    shape = *model.output_shapes[l][:-1], f[1][1] - f[1][0]
                    # f_size = max(f_size, cal_tensor_size(shape))
                    f_size += cal_tensor_size(shape)  # 多向发送是应该算发送的最大值还是发送总量？
            forward_workers[w].append(f_size)
    show_workers_transmission_size(forward_workers)


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
            recv_input[device][-1].append(x[..., ri[0]:ri[1]])  # cut from the last_array dimension of 4D input

        results = []
        threads = []
        for w in range(n_device):
            threads.append(threading.Thread(target=execute, args=[task_queue[w], recv_input[w], w, results]))
        for worker in threads:
            worker.start()
        for worker in threads:
            worker.join()
        results.sort(key=lambda item: item[0])
        result = torch.cat([r[1] for r in results], -1)
        # result = results[0][1]
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
        # send subtasks
        try:
            send_tasks = [async_send_data(m.worker_sockets[device], eus) for device, eus in enumerate(execution_units)]
            loop.run_until_complete(asyncio.gather(*send_tasks))
        except Exception as e:
            print(f'Error occurred when send subtasks:\n{e}')
        time.sleep(1)

        # send initial inputs
        start = time.time()  # worker接受所有tasks需要花很多时间
        try:
            send_tasks = [async_send_data(m.worker_sockets[device], first_input) for device, first_input in
                          enumerate(first_inputs)]
            loop.run_until_complete(asyncio.gather(*send_tasks))
        except Exception as e:
            print(f'Error occurred when send required input to workers:\n{e}')

        try:
            # data = recv_data(m.worker_sockets[0])  # 只收concat之后来自一个worker的output

            recv_tasks = [async_recv_data(sock) for sock in m.worker_sockets]
            results = loop.run_until_complete(asyncio.gather(*recv_tasks))
            consumption = time.time() - start
            results.sort(key=lambda item: item[0])
            data = torch.cat([r[1] for r in results], -1)
            print(torch.allclose(outputs[-1], data))
            print(f'Recv final result in {consumption}s')
        except timeout:
            print('Recv results time out!')
        except Exception as e:
            print(f'Error occurred when recv results:\n{e}')

        # 接受workers生成的计算区间：并尝试绘图
        recvs = None
        accum_time = None
        try:
            recv_tasks = [async_recv_data(sock) for sock in m.worker_sockets]
            recvs = loop.run_until_complete(asyncio.gather(*recv_tasks))  # execute_intervals of workers
            accum_time = [np.asarray(intervals) for intervals in recvs]
            show_time_intervals(recvs)

        except timeout:
            print('Recv intervals time out!')
        except Exception as e:
            print(f'Error occurred when recv intervals:\n{e}')

        ### replay the execution process:
        print("----------------------Process Replay----------------------")
        execution_workers = []
        gap_workers = []
        sums_workers = []
        for w in range(n_device):
            recv = recvs[w]
            num_tasks = len(execution_units[w])
            execution_times = []
            gap = []
            for t in range(num_tasks):
                execution_times.append(recv[t * 2])
                if t * 2 + 1 >= len(recv):
                    break
                gap.append(recv[t * 2 + 1])
            execution_times = np.asarray(execution_times)
            sums_workers.append(execution_times.sum())
            execution_idx10 = np.argsort(execution_times)[::-1][:10]
            gap = np.asarray(gap)
            gap_idx10 = np.argsort(gap)[::-1][:10]
            # print(execution_idx10)
            task_ids = [execution_units[w][i].layer_num for i in execution_idx10]
            execution_workers.append(list(zip(task_ids, execution_times[execution_idx10])))
            task_ids = [execution_units[w][i].layer_num for i in gap_idx10]
            gap_workers.append(list(zip(task_ids, gap[gap_idx10])))

        for w, exe_sum in enumerate(sums_workers):
            print(f'worker {w + 1} execution sum: {exe_sum}')
        # show the most time-consuming execution and gap intervals in the process
        print("The most time-consuming tasks with layer number")
        for w, execution_worker in enumerate(execution_workers):
            print(f'{w + 1}: {execution_worker}')
        print("The most time-consuming gap with last_array layer number")
        for w, gap_worker in enumerate(gap_workers):
            print(f'{w + 1}: {gap_worker}')

        print('Records of all subtasks')
        for w, intervals in enumerate(accum_time):
            records = []
            layers_id = [t.layer_num for t in execution_units[w]]
            intervals -= intervals[0]
            for i, layer in enumerate(layers_id):
                records.append((layer, intervals[i * 2]))
                records.append((layer, intervals[i * 2 + 1]))
            print(f'{w}: {records}')