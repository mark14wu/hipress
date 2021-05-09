import math
import os
import time

import horovod.mxnet as hvd
import mxnet as mx

import socket


if __name__ == "__main__":
    total_size = 2**27
    root_rank = 0
    recv = mx.nd.zeros(shape=(total_size))    
    send = mx.nd.ones(shape=(total_size))
    dep = mx.nd.zeros(shape=(16))
    times = 10
    hvd.init()
    # btic = time.time()
    # for i in range(times):
    #     hvd.gather_(recv, recv, dep, root_rank, num_elem=total_size//4)
    #     # hvd.gather_(recv, recv[:total_size//2], dep, root_rank)
    # recv.wait_to_read()
    # if hvd.rank() == root_rank:
    #     print('gather time:', (time.time() - btic)/times)

    btic = time.time()
    for i in range(times):
        # hvd.broadcast_(recv, root_rank, total_size//2)
        hvd.broadcast_(recv[:total_size//2], root_rank)
    recv.wait_to_read()
    if hvd.rank() == root_rank:
        print('bcast time:', (time.time() - btic)/times)