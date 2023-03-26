from worker_udp import *
from worker_udp import __subnet__, __port__
from comm import __buffer__, __timeout__


class Master(Worker):
    # The master node should:
    # 1. accept DNN inference request from mobile with input
    #   - a thread to keep listening the request
    #   - a queue to store multiple request (temporarily unnecessary)
    # 2. divide DNN inference task into several execution units
    # 3. forward and deploy the sub-tasks to other worker nodes
    #   - record the available worker nodes (ip)
    #   - a task queue to store and execute the allocated tasks
    # 4. execute its own units
    # 5. send back the results to the mobile
    def __init__(self):
        super().__init__()

    def recv_message(self, message: bytes, queue: SimpleQueue, stop):
        msg_len = len(message)
        while True:
            print("trying receiving!")
            try:
                msg, addr = self.recv_socket.recvfrom(msg_len)
                # print(message)
                print(f'{msg} from {addr}')
                # if msg == message:
                queue.put((msg, addr))
            except timeout:
                break
            except Exception as e:
                print(e)
            if stop():
                break

    def master_init(self):  # receive "available" message from workers and give back the list of all workers
        message_queue = SimpleQueue()
        print('Recv registers from workers...')
        stop_thread = False
        recv_thread = threading.Thread(target=self.recv_message,
                                       args=[b'available', message_queue, lambda: stop_thread])
        recv_thread.start()
        # start = time.time()
        time.sleep(__timeout__)
        stop_thread = True
        recv_thread.join()  # waiting for recv the register message from workers
        # print(time.time() - start)
        # stop_thread = True

        if message_queue.empty():
            print('No registration from workers')
            return False
        self.workers_config = {0: master_ip}
        n_worker = 1
        while not message_queue.empty():
            msg, worker_addr = message_queue.get()
            if msg == b'available':
                self.workers_config[n_worker] = worker_addr[0]
                n_worker += 1
        print(f"Recv registrations from {n_worker - 1} workers")
        print("Sending workers' config to all workers...")
        self.n_workers = n_worker
        serialized = pickle.dumps((self.n_workers, self.workers_config))
        for i in range(1, self.n_workers):  # send workers_config to every worker
            addr = (self.workers_config[i], __port__)
            self.send_socket.sendto(serialized, addr)
        message_queue = SimpleQueue()
        stop_thread = False
        recv_thread = threading.Thread(target=self.recv_message, args=[b'OK', message_queue, lambda: stop_thread])
        recv_thread.start()
        time.sleep(__timeout__)
        stop_thread = True
        recv_thread.join()
        # stop_thread = True

        n_reply = 0
        while not message_queue.empty():
            msg, addr = message_queue.get()
            if msg == b'OK':
                n_reply += 1
        # print(n_reply, self.n_workers)
        if n_reply == self.n_workers - 1:
        # if message_queue.qsize() == self.n_workers - 1:
            print('All workers are ready!')
            return True
        print('Some workers fail to reply')
        return False


if __name__ == '__main__':
    master = Master()
    ok = master.master_init()
    print(ok)
