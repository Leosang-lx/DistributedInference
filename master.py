import sys
import threading
import time

import torch

from comm import async_send_data, recv_data, send_tensor, recv_tensor, send_data
from worker import master_ip, __subnet__, __port__
from util import *
from queue import SimpleQueue


# use laptop as master
# consider only one DNN
def recv_message(conn: socket, recv_queue: SimpleQueue):
    try:
        msg = recv_data(conn)
        return msg
    except timeout:
        return None
    except Exception as e:
        print(e)
        return None


class Master:
    def __init__(self):
        # self.ip = get_ip_addr(__subnet__)
        # if self.ip != master_ip:
        #     print('Wrong master ip!')
        #     return
        self.ip = '192.168.1.101'  # use laptop as master
        self.server_socket = create_server((self.ip, __port__), family=AF_INET)
        self.server_socket.settimeout(8)
        self.worker_sockets = []
        self.inference_requests = SimpleQueue()

        # accept connections from workers
        print(f'Master ip is {self.ip}, recv connections from workers...')
        available_workers = []
        stop_thread = False
        recv_thread = threading.Thread(target=accept_connection,
                                       args=[self.server_socket, available_workers, lambda: stop_thread])
        recv_thread.start()
        time.sleep(10)
        stop_thread = True
        recv_thread.join()

        #
        self.no_workers = []
        worker_sockets = []
        for conn, addr in available_workers:
            # conn.settimeout(5)
            # ip, port = addr
            if addr not in self.no_workers:  # first time to meet the addr
                self.no_workers.append(addr)
                worker_sockets.append(conn)
            else:
                conn.close()
        self.worker_sockets = worker_sockets
        self.n_workers = len(self.no_workers)
        print(f'Recv connections from {self.n_workers} workers')
        for conn in worker_sockets:
            send_data(conn, self.no_workers)

        recv_queue = SimpleQueue()  # 接收workers确认与所有其他workers完成建立连接的消息
        for conn in self.worker_sockets:
            recv_thread = threading.Thread(target=put_recv_data, args=[conn, recv_queue])
            recv_thread.start()
            recv_thread.join()
        if self.n_workers != 0 and recv_queue.qsize() == self.n_workers:
            print('Successfully initialized!')
        else:
            print('Fail to initialize')

        self.result_list = []
        self.recv_threads = []

    def recv_results(self, recv_socket, result_list: list, id: int):
        while True:
            try:
                data = recv_data(recv_socket)
                result_list.append(data)
                print('Recv from worker', id)
            except timeout:
                print('Recv results time out!')
            except Exception as e:
                print(e)
            time.sleep(0.1)

    def start(self):
        """
        基于收到的input shape，对模型每一层的output进行拆分，计算每个子任务对应的input range，再根据子任务间的input range和
        output partition计算每一个子任务的forwarding，封装为子任务之后，和分割后的input传输给对应设备
        :return:None
        """
        print('Start recv threads')
        # for id, conn in enumerate(self.worker_sockets):
        #     recv_thread = threading.Thread(target=self.recv_results, args=[conn, self.result_list, id])
        #     self.recv_threads.append(recv_thread)
        # for thread in self.recv_threads:
        #     thread.start()
        # for thread in self.recv_threads:
        #     thread.join()



if __name__ == '__main__':
    master = Master()
    master.start()
    time.sleep(1)
    sys.exit(0)
