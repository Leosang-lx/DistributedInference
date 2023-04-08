from queue import SimpleQueue
from subprocess import getstatusoutput
from socket import *
import torch
from models.googlenet import GoogLeNet

from comm import *


# def union_interval(intervals):  # 只考虑单个最大区间
#     lefts = [interval[0] for interval in intervals]
#     rights = [interval[1] for interval in intervals]
#     return min(lefts), max(rights)


def get_intersection(interval1, interval2):
    left_max = max(interval1[0], interval2[0])
    right_min = min(interval1[1], interval2[1])
    if left_max < right_min:  # overlap
        return left_max, right_min
    else:
        return None


def get_union(interval1, interval2):
    left_max = max(interval1[0], interval2[0])
    right_min = min(interval1[1], interval2[1])
    if left_max <= right_min:  # overlap
        return min(interval1[0], interval2[0]), max(interval1[1], interval2[1])
    else:
        return None


def get_set_union(intervals):  # 考虑多个区间取并集后仍有多个区间
    intervals.sort(key=lambda range_item: range_item[0])
    i, j = 0, len(intervals) - 1
    while True:
        if i == j:
            break
        range1, range2 = intervals.pop(i), intervals.pop(i)
        union = get_union(range1, range2)
        if union is not None:
            intervals.insert(i, union)
        else:
            intervals.insert(i, range1)
            intervals.insert(i + 1, range2)
            i = i + 1
        j = len(intervals) - 1
    return intervals


def input_satisfactory(required_input: tuple, recv_list: list):  # 判断required_input是否满足
    """
    judge whether the required input of sub-task is satisfied with recv inputs
    :param required_input: a tuple (dependent layers in list, required input range)
    :param recv_list: recv input stored by layers
    :return: return the required concat input else None
    """
    dependent_layers, input_range = required_input
    # print(required_input)
    if len(dependent_layers) > 1:  # concat output from execution units of several layers
        collect = []
        for i, dl in enumerate(dependent_layers):
            outputs = recv_list[dl]
            if len(outputs) == input_range[i]:
                outputs.sort(key=lambda x: x[0][0])
                outputs = [data[1] for data in outputs]
                collect.append(torch.cat(outputs, -1))  # item in recv_list is (global range, data)
            else:
                return None
        return collect

    else:  # collect paddings len(dependent_layers) == 1 or == 0
        c, d = input_range
        if len(dependent_layers) == 0:  # input of the first layer
            if len(recv_list[-1]) == 0:
                return None
            return recv_list[-1][0]
        else:
            input_list = recv_list[dependent_layers[0]]
        input_list.sort(key=lambda item: item[0][0])
        collect = []
        for interval, data in input_list:
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
            # print(torch.concat(collect, dim=-1).shape)
            return torch.concat(collect, dim=-1)
        return None


def get_ip_addr(__subnet__: str):  # get ip by the prefix of __subnet__
    status, ip_addr = getstatusoutput(f'ifconfig | grep "{__subnet__}" | awk \'{{print $2}}\'')
    if status == 0:
        return ip_addr
    return None


def connect_to_other(ip_list: list, port: int, socket_list: list, self_ip):
    try:
        for i, worker_ip in enumerate(ip_list):
            if worker_ip != self_ip:
                addr = worker_ip, port
                conn = create_connection(addr, timeout=5)
                print(f'Connected to {worker_ip}')
                socket_list.append((conn, addr[0]))
    except timeout:
        print('Create connections timeout')
    except Exception as e:
        print(e)


# def connect_to_other_local(port_list: list, socket_list: list, self_ip):
#     try:
#         for i, port in enumerate(port_list):
#             if worker_ip != self_ip:
#                 addr = worker_ip, port
#                 conn = create_connection(addr, timeout=5)
#                 print(f'Connected to {worker_ip}')
#                 socket_list.append((conn, addr[0]))
#     except timeout:
#         print('Create connections timeout')
#     except Exception as e:
#         print(e)


def accept_connection(server_socket, recv_list: list, stop):
    while True:
        if stop():
            break
        try:
            conn, addr = server_socket.accept()
            recv_list.append((conn, addr[0]))  # only ip
            print(f'Recv connection from {addr}')
        except timeout:
            continue
        except Exception as e:
            print(e)


def put_recv_data(conn: socket, recv_queue: SimpleQueue):
    try:
        data = recv_data(conn)
        recv_queue.put(data)
    except timeout:
        print('Recv data time out')
    except Exception as e:
        print(e)


def load_model(model_name: str, dict_path=None, device='cpu'):
    model_name = model_name.lower()
    if model_name == 'googlenet':
        model = GoogLeNet()

    else:
        raise Exception('Invalid model name!')
    model.to(device)
    if dict_path is not None:
        model.load_state_dict(torch.load(dict_path, map_location=device))
    return model


def product(num_list):
    res = 1
    for i in num_list:
        res *= i
    return res


def cal_tensor_size(tensor_shape, fix=False):
    size = product(tensor_shape) * 4
    if fix:
        size += 48
    return size
