import asyncio
import socket
import threading
import time

import select

from util import *
from comm import *
from ExecutionUnit import ExecutionUnit

__subnet__ = '192.168.1'
__port__ = 49999
master_ip = '192.168.1.101'


class Worker:
    # The worker node should:
    # 1. accept the sub-tasks from master node
    #   - a task queue to store and execute the allocated tasks
    # 2. execute the sub-tasks with required input
    # 3. transfer required input of other devices
    #   - a thread to accept required input from other devices
    #   - a thread to forward required input of other devices
    def __init__(self):
        self.number = None
        # self.ip = None
        self.ip = get_ip_addr(__subnet__)
        if self.ip is None:
            print('Fail to acquire ip, try again...')
            self.ip = get_ip_addr(__subnet__)
            if self.ip is None:
                print('Fail to acquire ip, exit')
                return
        print(f'ip_addr: {self.ip}')
        print(f'Connected to master(ip: {master_ip})...')

        master_socket = None
        try:
            master_socket = create_connection(address=(master_ip, __port__))  # , timeout=15)
        except Exception as e:
            print('When connect to master, error occurs:\n' + e.__str__())
        if master_socket is not None:
            self.master_socket = master_socket
        else:
            print('Fail to connect to master.')

        self.no_workers = None
        self.n_worker = 0
        try:
            no_workers = recv_data(self.master_socket)
            if isinstance(no_workers, list):
                print('Recv from master')
                self.no_workers = no_workers
                self.n_worker = len(no_workers)
            else:
                print('Recv wrong data')
        except timeout:
            print('Recv time out')
        except Exception as e:
            print(e)

        self.worker_no = {}
        for i, worker_ip in enumerate(self.no_workers):
            self.worker_no[worker_ip] = i
        self.number = self.worker_no[self.ip]

        # build connections between workers
        self.server_socket = None
        stop_thread = False
        recv_sockets = []  # 接受连接的socket
        send_sockets = []  # 主动连接的socket
        self.server_socket = create_server((self.ip, __port__), family=AF_INET)
        self.server_socket.settimeout(8)
        recv_socket_thread = threading.Thread(target=accept_connection,
                                              args=[self.server_socket, recv_sockets, lambda: stop_thread])
        recv_socket_thread.start()

        time.sleep(3)

        send_socket_thread = threading.Thread(target=connect_to_other, args=[self.no_workers, __port__,
                                                                             send_sockets, self.ip])
        send_socket_thread.start()

        time.sleep(5)
        stop_thread = True
        send_socket_thread.join()
        recv_socket_thread.join()
        self.recv_list = [sock[0] for sock in recv_sockets]

        print(len(recv_sockets), len(send_sockets))
        if len(recv_sockets) == len(send_sockets) == self.n_worker - 1:
            self.recv_sockets = [None for _ in range(self.n_worker)]
            for conn, addr in recv_sockets:
                worker_id = self.worker_no[addr]
                self.recv_sockets[worker_id] = conn
            self.send_sockets = [None for _ in range(self.n_worker)]
            for conn, addr in send_sockets:
                conn.settimeout(10)
                worker_id = self.worker_no[addr]
                # conn.setblocking(False)  # send非阻塞。。。？
                self.send_sockets[worker_id] = conn
        none1 = self.send_sockets.count(None)  # one None in list
        none2 = self.recv_sockets.count(None)  # one None in list

        if none1 == none2 == 1:
            send_data(self.master_socket, 'Roger')
            print('Successfully initialized!')
        else:
            print("Fail to recv workers' config")
        self.model = None
        self.recv_inputs = None
        self.task_queue = SimpleQueue()
        self.execute_queue = SimpleQueue()

    async def start(self):
        """
        开启多个线程，分别用于接受子任务，接受输入，以及处理子任务
        参控net_analysis.py的仿真
        :return:None
        """

        print('Loading DNN model...')
        self.model = load_model('googlenet')
        self.recv_inputs = [[] for _ in range(self.model.depth)]

        # recv_threads = []
        # for conn in self.recv_sockets:
        #     if conn is None:
        #         continue
        #     recv_thread = threading.Thread(target=self.recv_input, args=[conn])
        #     recv_thread.start()
        #     recv_threads.append(recv_thread)

        input_processing_thread = threading.Thread(target=self.get_available_task)
        input_processing_thread.start()

        print('Waiting for subtasks and inputs')
        try:
            subtasks = recv_data(self.master_socket)
            for t in subtasks:
                self.task_queue.put(t)
            print('Recv subtasks')
            self.recv_inputs[-1].append(recv_data(self.master_socket))
            print('Recv input')
        except timeout:
            print('Time out for recv subtasks')
        except Exception as e:
            print(e)

        while True:
            # self.get_available_task()

            ready = select.select(self.recv_list, [], [], 0.001)
            read_ready = ready[0]
            for sock in read_ready:
                source, data = recv_data(sock)
                layer_no, input_range = source
                print(f'Recv layer {layer_no} output')
                self.recv_inputs[layer_no].append((input_range, data))

            try:
                from_queue = self.execute_queue.get(timeout=0.01)
            except Exception as e:
                continue

            if from_queue is not None:
                task, args = from_queue
                # assert isinstance(task, ExecutionUnit)
                print(f'Execute task {task.layer_num}')
                output = task.execute(*args)
                # if task.forwarding == 1:
                #     try:
                #         send_data(self.master_socket, output)
                #         print('Send final output to master')
                #     except Exception as e:
                #         print(f'Error occurs when sending final output to master: {e}')
                # else:
                if task.layer_num == self.model.depth - 1:
                    try:
                        # asyncio.create_task(async_send_data(self.master_socket, output))
                        await async_send_data(self.master_socket, output)
                    except Exception as e:
                        print(f'Error occurs when sending final output to master: {e}')
                for f in task.forwarding:
                    if len(f) == 2:  # (to_device, interval)
                        to_device, interval = f
                        l, r = interval
                    else:  # len = 3: (to_device, (interval[0] - partition[i], interval[1] - partition[i]), partition[i]))
                        to_device, interval, left = f
                        l, r = interval[0] + left, interval[1] + left
                    if to_device == self.number:
                        self.recv_inputs[task.layer_num].append(((l, r), output[..., interval[0]:interval[1]]))
                    else:
                        try:
                            # async_task = asyncio.create_task(async_send_data(self.send_sockets[to_device], (
                            # (task.layer_num, (l, r)), output[..., interval[0]:interval[1]])))
                            await async_send_data(self.send_sockets[to_device], (
                            (task.layer_num, (l, r)), output[..., interval[0]:interval[1]]))
                            # await async_task
                        except Exception as e:
                            print(f'Error occurs when sending output: {e}')


    # def recv_input(self, recv_socket):  # start n_workers-1 thread to recv input from other workers
    #     while True:
    #         source, data = recv_data(recv_socket)
    #         layer_no, input_range = source
    #         print(f'Recv layer {layer_no} output')
    #         self.recv_inputs[layer_no].append((input_range, data))
    #         time.sleep(0.1)

    def get_available_task(self):

        task = None
        while True:
            if task is None:
                # assert isinstance(task, ExecutionUnit)
                task = self.task_queue.get()  # block when queue is empty
            required_input = input_satisfactory(task.required_input, self.recv_inputs)
            if required_input is None:
                # print('Not ready')
                time.sleep(0.1)
                continue
                # return
            else:

                print(f'Task {task.layer_num} is ready')
                args = [required_input]
                if task.operator['type'] == 'basicConv':
                    args.append(self.model.layers[task.layer_num].conv.weight)
                self.execute_queue.put((task, args))
                task = None


if __name__ == '__main__':
    worker = Worker()
    loop = asyncio.get_event_loop()
    asyncio.run(worker.start())
