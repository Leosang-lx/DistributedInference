import concurrent.futures
import random
import time
from multiprocessing import Queue, Pool, cpu_count

def read(q):
    print(('Get %s from queue.' % q))
    time.sleep(random.random())


def main():
    futures = set()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for q in (chr(ord('A')+i) for i in range(26)):
            future = executor.submit(read, q)
            futures.add(future)
    try:
        for future in concurrent.futures.as_completed(futures):
            err = future.exception()
            if err is not None:
                raise err
            else:
                print(future.result())
    except KeyboardInterrupt:
        print('Stopped by hand')

if __name__ == '__main__':
    print(f'{cpu_count()} cpus')