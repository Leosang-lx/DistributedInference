import random
import time
from multiprocessing import Queue
import threading
from queue import SimpleQueue
from comm import *
from comm import __buffer__, __timeout__
import argparse

__subnet__ = '192.168.1'  # using hotpot
__port__ = 49999
master_ip = '192.168.1.1'
# ps = argparse.ArgumentParser()
# ps.add_argument('port', '-p', )


def get_ip_addr():  # get ip by the prefix of __subnet__
    status, ip_addr = getstatusoutput(f'ifconfig | grep "{__subnet__}" | awk \'{{print $2}}\'')
    if status == 0:
        return ip_addr
    return None


def gen_socket(bind_ip=None, port=None):
    soc = socket(AF_INET, SOCK_STREAM)
    while True:
        try:
            if bind_ip is not None:
                i = port if port else random.randint(20000, 65536)
                soc.bind((bind_ip, i))
            else:
                return soc
            break
        except Exception as e:
            print(f'{e}: Fail to bind ip {bind_ip} with port {port}')
            # i = random.randint(20000, 65536)
    soc.settimeout(__timeout__)
    return soc


def get_ping_result(dest:str, ping_cnt: int, timeout: int, queue: SimpleQueue):
    if ping(dest, ping_cnt, timeout):
        queue.put(dest)
    # queue.put(ping(dest, ping_cnt, timeout))


class Worker:
    # The worker node should:
    # 1. accept the sub-tasks from master node
    #   - a task queue to store and execute the allocated tasks
    # 2. execute the sub-tasks with required input
    # 3. transfer required input of other devices
    #   - a thread to accept required input from other devices
    #   - a thread to forward required input of other devices
    def __init__(self):
        self.no_device = None
        self.ip = None
        self.n_workers = None
        self.workers_config = None  # store the matching of No. and ip of workers, the first one is the master
        self.recv_socket = None
        self.send_socket = None

        # self.sockets = []
        # self.task_accept_thread = threading.Thread()
        self.task_queue = Queue()
        self.init()

        # self.send_thread_pool =

    def worker_init(self) -> bool:  # register worker to the master node, return value is the status of init
        master_addr = (master_ip, __port__)
        print('Register to Master')
        self.send_socket.sendto(b'available', master_addr)
        time.sleep(1)
        try:
            serialized = self.recv_socket.recv(__buffer__)
            self.n_workers, self.workers_config = pickle.loads(serialized)  # (int, dict)
        except Exception as e:
            print(e)
            return False
        print('Recv workers_config from Master')
        q = SimpleQueue()
        print('Ping to all workers...')
        for i in range(1, self.n_workers):
            thread = threading.Thread(target=get_ping_result, args=[self.workers_config[i], 1, 1, q])
            thread.start()
            thread.join()
        if q.qsize() == self.n_workers - 1:
            self.send_socket.sendto(b'OK', master_addr)
            return True
        return False

    def init(self):
        # local_ip = get_ip_addr()  # get local ip
        local_ip = '192.168.137.1'
        if local_ip is None:
            print('Device is not connected to given __subnet__')
        else:
            print(f'ip: {local_ip}')
            self.ip = local_ip
            if self.recv_socket is None:
                self.recv_socket = gen_socket(local_ip, __port__)
            if self.send_socket is None:
                self.send_socket = gen_socket()


if __name__ == '__main__':
    # ps = argparse.ArgumentParser()
    # ps.add_argument()
    worker = Worker()
    ok = worker.worker_init()
    print(ok)
