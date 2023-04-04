import pickle
from socket import *
import struct
from torch import Tensor
from subprocess import getstatusoutput


__buffer__ = 65535
__timeout__ = 10


def ping(dest: str, ping_cnt: int, timeout: int):  # '1' means connected, else 0
    assert 0 < ping_cnt < 10
    status, output = getstatusoutput(
        f"ping {dest} -c {ping_cnt} -w {timeout} | tail -2 | head -1 | awk {{'print $4'}}")  # send one packet and check the recv packet
    print(f'ping result {output}')
    if status == 0 and int(output) > 0:
        return True  # ping得通
    else:
        return False


data_header_format = 'I'
data_header_size = struct.calcsize(data_header_format)


def send_data(send_socket: socket, data):
    serialized = pickle.dumps(data)
    data_size = len(serialized)
    header = struct.pack(data_header_format, data_size)
    res = send_socket.sendall(header + serialized)
    return res

async def async_send_data(send_socket: socket, data):
    serialized = pickle.dumps(data)
    data_size = len(serialized)
    header = struct.pack(data_header_format, data_size)
    res = send_socket.sendall(header + serialized)
    return res


def recv_data(recv_socket: socket):
    msg = recv_socket.recv(data_header_size)
    header = struct.unpack(data_header_format, msg)
    data_size = header[0]
    data = b''
    while data_size:
        recv = recv_socket.recv(min(4096, data_size))
        data += recv
        data_size -= len(recv)
    return pickle.loads(data)


__format__ = 'IHHH'  # header: (data size, layer num, left range, right range]
__size__ = struct.calcsize(__format__)


def send_tensor(send_socket: socket, data: Tensor, layer_num: int, data_range: tuple[int, int]):
    """
    send the padding data of intermediate result of certain layer to other device
    :param send_socket: the socket of UDP protocol to send data
    :param data: the (Tensor type) data to send
    :param layer_num: the intermediate result of which layer
    :param data_range: the corresponding range of data to the layer output
    :return: the sending status
    """
    serialized = pickle.dumps(data)
    data_size = len(serialized)
    header = struct.pack(__format__, data_size, layer_num, *data_range)
    res = send_socket.sendall(header + serialized)
    return res


async def async_send_tensor(send_socket: socket, data: Tensor, layer_num: int, data_range: tuple[int, int]):
    print(f'send output of layer {layer_num}')
    serialized = pickle.dumps(data)
    data_size = len(serialized)
    header = struct.pack(__format__, data_size, layer_num, *data_range)
    res = send_socket.sendall(header + serialized)
    return res


def recv_tensor(recv_socket: socket):
    data = recv_socket.recv(__size__)
    header = struct.unpack(__format__, data)
    data_size = header[0]
    data = b''
    while data_size:
        recv = recv_socket.recv(min(4096, data_size))
        data += recv
        data_size -= len(recv)
    return (header[1], (header[2], header[3])), pickle.loads(data)  # (layer_num, range) data


async def async_recv_tensor(recv_socket: socket):
    data = recv_socket.recv(__size__)
    header = struct.unpack(__format__, data)
    data_size = header[0]
    data = b''
    while data_size:
        recv = recv_socket.recv(min(4096, data_size))
        data += recv
        data_size -= len(recv)
    return (header[1], (header[2], header[3])), pickle.loads(data)  # (layer_num, range) data
