import asyncio
import sys
import threading
import torch.nn as nn
import time
from ExecutionUnit import ExecutionUnit
from models.googlenet import BasicConv2d
from paint import *
from util import *
from net_analysis import next_to_last, topology_DAG, cal_output_shape, workload_split, output_input

model_name = 'vgg16'
model = load_model(model_name)
# layers = [
#     model.conv1,  # 0
#     model.maxpool1,  # 1
#     model.conv2,  # 2
#     model.conv3,  # 3
#     model.maxpool2,  # 4
#     model.inception3a.branch1,  # 5
#     *model.inception3a.branch2,  # 67
#     *model.inception3a.branch3,  # 89
#     *model.inception3a.branch4,  # 1011
#     # concat 70
#     model.inception3b.branch1,  # 12
#     *model.inception3b.branch2,  # 1314
#     *model.inception3b.branch3,  # 1516
#     *model.inception3b.branch4,  # 1718
#
#     model.maxpool3,  # 19
#     model.inception4a.branch1,  # 20
#     *model.inception4a.branch2,  # 2122
#     *model.inception4a.branch3,  # 2324
#     *model.inception4a.branch4,  # 2526
#
#     model.inception4b.branch1,  # 27
#     *model.inception4b.branch2,  # 2829
#     *model.inception4b.branch3,  # 3031
#     *model.inception4b.branch4,  # 3233
#
#     model.inception4c.branch1,  # 34
#     *model.inception4c.branch2,  # 3536
#     *model.inception4c.branch3,  # 3738
#     *model.inception4c.branch4,  # 3940
#
#     model.inception4d.branch1,  # 41
#     *model.inception4d.branch2,  # 4243
#     *model.inception4d.branch3,  # 4445
#     *model.inception4d.branch4,  # 4647
#
#     model.inception4e.branch1,  # 48
#     *model.inception4e.branch2,  # 4950
#     *model.inception4e.branch3,  # 5152
#     *model.inception4e.branch4,  # 5354
#
#     model.maxpool4,  # 55
#     model.inception5a.branch1,  # 56
#     *model.inception5a.branch2,  # 5758
#     *model.inception5a.branch3,  # 5960
#     *model.inception5a.branch4,  # 6162
#
#     model.inception5b.branch1,  # 63
#     *model.inception5b.branch2,  # 6465
#     *model.inception5b.branch3,  # 6667
#     *model.inception5b.branch4,  # 6869
#
#     # model.avgpool,  # 70  avgpool with output_shape (1, 1) means kernel_size = input_shape
#     # model.dropout,  # 71
#     # model.fc,  # 72
#     'concat',  # 70
#     'concat',  # 71
#     'concat',  # 72
#     'concat',  # 73
#     'concat',  # 74
#     'concat',  # 75
#     'concat',  # 76
#     'concat',  # 77
#     'concat',  # 78
# ]

last_concat = 78

# next_array = [1, 2, 3, 4, [5, 6, 8, 10], 70, 7, 70, 9, 70, 11, 70, 71, 14, 71, 16, 71, 18, 71, [20, 21, 23, 25], 72, 22, 72,
#         24, 72, 26, 72, 73, 29, 73, 31, 73, 33,
#         73, 74, 36, 74, 38, 74, 40, 74, 75, 43, 75, 45, 75, 47, 75, 76, 50, 76, 52, 76, 54, 76, [56, 57, 59, 61], 77,
#         58, 77, 60, 77, 62, 77, 78, 65, 78,
#         67, 78, 69, 78, [12, 13, 15, 17], 19, [27, 28, 30, 32], [34, 35, 37, 39], [41, 42, 44, 46],
#         [48, 49, 51, 53], 55, [63, 64, 66, 68], []]

# print(next_array[63])
# print(len(next_array))
# sys.exit(0)

def translate_next_array(next_array):
    for i in range(len(next_array)):  # 将next数组内的单个元素处理为长度为1的列表
        if not isinstance(next_array[i], list):
            next_array[i] = [next_array[i]]

# last_array = next_to_last(next_array)
# print(last_array)
#
# topology_list = topology_DAG(next_array, last_array)
#
# model.output_shapes = cal_output_shape(model, topology_list, last_array)
#
# n_device = 2
# output_partitions = workload_split(model.output_shapes, n_device)


def gen_inputDependency(layer_list, topology_list, output_partitions, last_array):
    not_concat = len(layer_list) - layer_list.count('concat')  # the number of layers that is not concat
    ids = [list() for _ in range(not_concat)]
    for l in topology_list:

        partitions = output_partitions[l]
        if layer_list[l] == 'concat':
            if l == last_concat:  # 如果这个concat层后面没有了，那就concat
                last_division_num = tuple(len(output_partitions[last_layer]) - 1 for last_layer in last_array[l])
                required_input = (last_array[l], last_division_num)
                layer_config = {'type': 'concat'}
                ids.append([(required_input, layer_config, [])])
            else:
                continue
        else:
            last = last_array[l]
            if len(last) == 1 and layer_list[last[0]] == 'concat':
                last = last_array[last[0]]  # if last layer is concat

            if l == 0:
                H = model.input_shape[-1]
            else:
                H = model.output_shapes[last[0]][-1]
            for i in range(len(partitions) - 1):
                # get output range
                output_range = partitions[i:i+2]  # [o_s, o_e)
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
                        padding = (upper_padding, bottom_padding, *layer.padding)  # consider the padding of two dimension is same
                        input_range = i_s, i_e
                    layer_config['padding'] = padding

                elif isinstance(layer, nn.MaxPool2d):
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
                    input_range = i_s, i_e
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

                required_input = last, input_range
                ids[l].append((required_input, layer_config, []))

    return ids


def gen_forwarding(layer_list, n_device: input, input_dependency, topology_list, next_array, output_partitions):
    for l in topology_list:
        partitions = output_partitions[l]
        nexts = next_array[l]
        if layer_list[l] == 'concat':
            continue
            # if l == last_concat:
            #     forwarding = [[] for _ in range(n_device)]
            #     for nl in next_array:
            #         for i, eu in enumerate(input_dependency[nl]):
            #             input_range = eu[0][1]
            #             forwarding[i].append(input_range)
            #     for to_device, f in enumerate(forwarding):  # d为设备号
            #         if len(f) > 0:  # 按照转发对应的设备号去重
            #             for interval in get_set_union(f):
            #                 input_dependency[l][0][2].append((to_device, interval))
            # else:
            #     continue

        else:

            if len(nexts) == 1 and layer_list[nexts[0]] == 'concat':
                if nexts[0] == 78:
                    for i in range(len(partitions) - 1):
                        input_dependency[l][i][2].append((0, (0, partitions[i + 1] - partitions[i]), partitions[i]))
                    continue
                else:
                    nexts = next_array[nexts[0]]  # concat the output from nexts before executing this task

            for lw in range(len(partitions) - 1):
                forwarding = [[] for _ in range(n_device)]
                for nl in nexts:
                    for nw, eu in enumerate(input_dependency[nl]):
                        _, input_range = eu[0]
                        overlap = get_intersection((partitions[lw], partitions[lw + 1]), input_range)
                        if overlap is not None:
                            forwarding[nw].append(overlap)
                for to_device, f in enumerate(forwarding):
                    if len(f) > 0:
                        for interval in get_set_union(f):
                            input_dependency[l][lw][2].append((to_device, (interval[0] - partitions[lw], interval[1] - partitions[lw]), partitions[lw]))


def gen_executionUnits(n_device: int, workload_partition, topology_list):
    device_group = [[] for _ in range(n_device)]
    for l, eus in enumerate(workload_partition):  # current layer
        for i, eu in enumerate(eus):
            device_group[i].append(
                ExecutionUnit(required_input=eu[0], operator=eu[1], forwarding=eu[2], layer_num=l, device_num=i))
    return device_group


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


def input_satisfactory2(required_input: tuple, recv_list: list):  # 判断required_input是否满足
    """
    judge whether the required input of sub-task is satisfied with recv inputs
    :param required_input: a tuple (dependent layers in list, required input range)
    :param recv_list: recv input stored by layers
    :return: return the required concat input else None
    """
    dependent_layers, input_range = required_input
    if len(dependent_layers) == 0:  # input of the first layer
        if len(recv_list[-1]) == 0:
            return None
        return recv_list[-1][0]
    # print(required_input)
    # if len(dependent_layers) > 1:  # concat output from execution units of several layers
    collects = []
    if len(dependent_layers) != len(input_range):
        for last_last in dependent_layers:
            c, d = input_range
            outputs = recv_list[last_last]
            outputs.sort(key=lambda x: x[0][0])
            collect = []
            for interval, data in outputs:
                a, b = interval
                if a <= c < b:
                    if d <= b:
                        collect.append(data[..., c - a:d - a])
                        c = d
                        break
                    else:
                        collect.append(data[..., c - a:])
                    c = b
            if c == d:
                collects.append(collect)
            else:
                return None
        if len(dependent_layers) == 1:
            return torch.cat(collects[0], -1)
        return torch.cat([torch.cat(concat, -1) for concat in collects], 1)

    else:  # concat layer
        for i, dl in enumerate(dependent_layers):
            outputs = recv_list[dl]
            if len(outputs) == input_range[i]:
                outputs.sort(key=lambda x: x[0][0])
                outputs = [data[1] for data in outputs]
                collects.append(torch.cat(outputs, -1))  # item in recv_list is (global range, data)
            else:
                return None
        return collects

    # else:  # collect paddings len(dependent_layers) == 1 or == 0
    #     c, d = input_range
    #     if len(dependent_layers) == 0:  # input of the first layer
    #         if len(recv_list[-1]) == 0:
    #             return None
    #         return recv_list[-1][0]
    #     else:
    #         input_list = recv_list[dependent_layers[0]]
    #     input_list.sort(key=lambda item: item[0][0])
    #     collect = []
    #     for interval, data in input_list:
    #         a, b = interval
    #         if a <= c < b:
    #             if d <= b:
    #                 collect.append(data[..., c - a:d - a])
    #                 c = d
    #                 break
    #             else:
    #                 collect.append(data[..., c - a:])
    #             c = b
    #     if c == d:
    #         # print(torch.concat(collect, dim=-1).shape)
    #         return torch.concat(collect, dim=-1)
    #     return None


def execute(tq: SimpleQueue, ri: list, worker_no: int, result_list: list):
    # global output
    print(f'this is worker {worker_no}')
    while not tq.empty():

        task = tq.get()
        layer_no = task.layer_num
        required_input = input_satisfactory2(task.required_input, ri)
        if layer_no == 0:
            dependent_layers, input_range = task.required_input
            right_input = torch.equal(required_input, x[..., input_range[0]:input_range[1]])
        start = time.time()
        while required_input is None:
            time.sleep(0.1)
            required_input = input_satisfactory2(task.required_input, ri)
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
        if not same:
            h = 1
        # same_shape = output.shape == correct.shape
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
        if len(task.forwarding) == 0:
            result_list.append((worker_no, output))
    # result_list.append((worker_no, output))


if __name__ == '__main__':
    model.input_shape = 3, 224, 224
    n_device = 3
    next_array = model.next
    translate_next_array(next_array)
    layers = model.layers
    layers_dependency = next_to_last(next_array)
    n_layers = len(layers_dependency)
    topology_layers = topology_DAG(next_array, layers_dependency)

    model.output_shapes = cal_output_shape(model, topology_layers, layers_dependency)

    partitions = workload_split(model.output_shapes, n_device)
    # partitions[1][1] = 74
    for i, partition in enumerate(partitions):
        print(f'{i}: {partition}')
    # print(partitions[73:])

    workload_dependency = gen_inputDependency(layers, topology_layers, partitions, layers_dependency)

    gen_forwarding(layers, n_device, workload_dependency, topology_layers, next_array, partitions)

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
    show_transmission_size(forward_workers)

    required_inputs = [execution_units[i][0].required_input[1] for i in range(n_device)]
    print(required_inputs)
    x = torch.randn((1, *model.input_shape))  # test input
    outputs = cal_output(topology_layers, layers_dependency, x)

    # right_output = model.conv1(x)
    # compt_outputs = []
    # first_eu = [execution_units[i][0] for i in range(n_device)]
    # ris = [x[..., eu.required_input[1][0]:eu.required_input[1][1]] for eu in first_eu]
    # for i in range(n_device):
    #     weight = model.conv1.conv.weight
    #     compt_outputs.append(first_eu[i].execute(ris[i], weight))
    # final_output = torch.cat(compt_outputs, dim=-1)
    # print(torch.equal(right_output, final_output))

    simulate = True
    if not simulate:
        sys.exit(0)
    local = False
    if local:
        # 在本地模拟DNN拆分子任务执行判断拆分是否正确
        task_queue = [SimpleQueue() for _ in range(n_device)]  # store all tasks
        # execute_queue = [SimpleQueue() for _ in range(n_device)]  # store input-ready tasks
        recv_input = [[[] for _ in range(n_layers + 1)] for _ in range(n_device)]  # item: (input, from_layer)

        for device, nl in enumerate(execution_units):  # put tasks into corresponding device queue
            for eu in nl:
                task_queue[device].put(eu)

        # how to store the recv input and how to judge required input is satisfied
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
            print(f'worker {w+1} execution sum: {exe_sum}')
        # show the most time-consuming execution and gap intervals in the process
        print("The most time-consuming tasks with layer number")
        for w, execution_worker in enumerate(execution_workers):
            print(f'{w+1}: {execution_worker}')
        print("The most time-consuming gap with last_array layer number")
        for w, gap_worker in enumerate(gap_workers):
            print(f'{w+1}: {gap_worker}')

        print('Records of all subtasks')
        for w, intervals in enumerate(accum_time):
            records = []
            layers_id = [t.layer_num for t in execution_units[w]]
            intervals -= intervals[0]
            for i, layer in enumerate(layers_id):
                records.append((layer, intervals[i * 2]))
                records.append((layer, intervals[i * 2 + 1]))
            print(f'{w}: {records}')



