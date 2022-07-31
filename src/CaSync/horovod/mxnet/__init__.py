# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from horovod.common import check_extension

check_extension('horovod.mxnet', 'HOROVOD_WITH_MXNET',
                __file__, 'mpi_lib')

from horovod.mxnet.mpi_ops import allgather, allgather_
from horovod.mxnet.mpi_ops import allreduce, allreduce_
from horovod.mxnet.mpi_ops import broadcast, broadcast_, mpibroadcast_
from horovod.mxnet.mpi_ops import gather_and_broadcast, gather_and_broadcast_
from horovod.mxnet.mpi_ops import gather_
from horovod.mxnet.mpi_ops import init, shutdown
from horovod.mxnet.mpi_ops import size, local_size, rank, local_rank
from horovod.mxnet.mpi_ops import mpi_threads_supported

from horovod.mxnet.gradcomp import GradientCompressEngine
from horovod.mxnet.gradcomp.compressor.dgc import DgcCompressor
from horovod.mxnet.gradcomp.compressor.topkadam import TopKAdamCompressor
from horovod.mxnet.gradcomp.compressor.powersgd import PowerSGDCompressor

import mxnet as mx
import types
import warnings
import math
import struct
import time
import socket
import sys
import logging
import functools

# This is where Horovod's DistributedOptimizer wrapper for MXNet goes
class DistributedOptimizer(mx.optimizer.Optimizer):
    def __init__(self, optimizer, comp_alg='', threshold=sys.maxsize, **kwargs):
        """
        initialize the horovod distributed optimizer

        Arguments:
            optimizer: a local optimizer like sgd
            comp_alg: str type, the name of gradient compression algorithm, 
                      choose from 'tbq, terngrad, ecq, mgc, dgc, graddrop, adacomp'
            threshold: if gradient size is greater than threshold, do compress
            **kwargs: the hyper parameter for chosen algorithm
                      for tbq:
                          threshold: 0.5
                      for terngrad:
                          bit_width: choose from 1 2 4 8
                      for ecq:
                          alpha: 0.2, beta: 0.9, bitwidth: 2
                      for dgc:
                          s_percent: default is 0.001,
                          sample_rate: default is 0.001
                      for graddrop:
                          drop_ratio: 0.999, sample_rate=1/1000
                      for adacomp:
                          T 50 or 500, default is 500
        """
        sparsity_algs = set(['dgc', 'graddrop', 'topkadam'])
        quantization_algs = set(['tbq', 'ecq', 'terngrad', 'mgc'])
        low_rank_decomp_algs = set(['powersgd'])
        optimizer_compression_algs = set(['topkadam'])
        other_algs = set(['adacomp'])
        supported_algs = functools.reduce(set.union, [
            sparsity_algs,
            quantization_algs,
            low_rank_decomp_algs,
            optimizer_compression_algs,
            other_algs
        ])
        self._optimizer = optimizer

        self._compress = None
        self._decompress = None
        self._residual = False

        self._comp_alg = comp_alg.lower()
        self._threshold = threshold
        self._bits_of_compressed_grad = None
        self._comp_parameters_map = dict()
        self._decomp_parameters_map = dict()

        self._sparsity = self._comp_alg in sparsity_algs
        self._quantization = self._comp_alg in quantization_algs
        self._low_rank_decomp = self._comp_alg in low_rank_decomp_algs
        self._do_compression = self._comp_alg in supported_algs
        self._optimizer_compression = self._comp_alg in optimizer_compression_algs
        self._compression_rate = 0.0

        self._comp_parameters_map = {'tbq':{}, 'terngrad':{}, 'ecq':{}, 'dgc':{}, 'graddrop':{}, 'adacomp':{}, 'mgc':{}}
        self._decomp_parameters_map = {'tbq':{}, 'terngrad':{}, 'ecq':{}, 'dgc':{}, 'graddrop':{}, 'adacomp':{}, 'mgc':{}}

        self._partition_byte = 4*1024*1024 # default partition size is 32MB
        self._comp_gpus = dict() # from keyid to partitioned compressed gradients on gpu
        self._comp_cpus = dict() # from keyid to partitioned compressed gradients on cpu
        self._resd_gpus = dict() # from keyid to partitioned residuals
        self._powersgd_M_gpus = dict() # used for powersgd only
        self._q_memory = dict()
        self._p_memory = dict()

        self._key_start = 1000

        self.engine = None

        print("init optimizer:", self._comp_alg, self._threshold, self._sparsity, self._do_compression)

        assert(isinstance(comp_alg, str))
        if self._comp_alg != '':
            if self._comp_alg == 'tbq':
                self._compress = mx.nd.contrib.zgc
                self._decompress = mx.nd.contrib.zgcr
                self._residual = True
                self._bits_of_compressed_grad = lambda x: 2*x
                self._comp_parameters_map['tbq']['threshold'] = 1
                self._decomp_parameters_map['tbq']['threshold'] = 1
                self._compression_rate = 2/32
            elif self._comp_alg == 'mgc':
                self._compress = mx.nd.contrib.mgc
                self._decompress = mx.nd.contrib.mgcr
                self._residual = True
                self._bits_of_compressed_grad = lambda x: 2*x
                self._comp_parameters_map['mgc']['threshold'] = 1
                self._decomp_parameters_map['mgc']['threshold'] = 1
                self._compression_rate = 2/32
            elif self._comp_alg == 'terngrad':
                self._compress = mx.nd.contrib.terngrad
                self._decompress = mx.nd.contrib.terngradr
                self._bits_of_compressed_grad = lambda x: 80+2*x
                self._comp_parameters_map['terngrad']['bitwidth'] = 2
                self._comp_parameters_map['terngrad']['random'] = 0
                self._decomp_parameters_map['terngrad']['bitwidth'] = 2
                self._compression_rate = 2/32
            elif self._comp_alg == 'ecq':
                self._compress = mx.nd.contrib.ecq_sgd
                self._decompress = mx.nd.contrib.ecq_sgd_r
                self._residual = True
                self._bits_of_compressed_grad = lambda x: 80+8*math.ceil(x/4)
                self._comp_parameters_map['ecq']['alpha'] = 0.2
                self._comp_parameters_map['ecq']['beta'] = 0.9
                self._comp_parameters_map['ecq']['bitwidth'] = 2
                self._compression_rate = 2/32
            elif self._comp_alg == 'dgc':                
                self._compress = mx.nd.contrib.dgc_new
                self._decompress = mx.nd.contrib.dgc_new_r
                self._bits_of_compressed_grad = lambda x: (2*math.ceil(x*self._s_percent)+1)*32 # in float                
                self._s_percent = kwargs['s_percent'] if 's_percent' in kwargs.keys() else 0.001                
                self._sample_rate = kwargs['sample_rate'] if 'sample_rate' in kwargs.keys() else 0.001
                self._comp_parameters_map['dgc']['s_percent'] = self._s_percent
                self._comp_parameters_map['dgc']['sample_rate'] = self._sample_rate
                self._param_tmp = self._s_percent
                self._compression_rate = lambda x: x*self._s_percent
                self.engine = GradientCompressEngine(
                    compressor=DgcCompressor(s_percent=self._s_percent, sample_rate=self._sample_rate)
                )
            elif self._comp_alg == 'graddrop':
                self._compress = mx.nd.contrib.scdgd
                self._decompress = mx.nd.contrib.scdgd_r
                self._residual = True
                self._drop_ratio = 0.999
                self._sample_rate = 0.001
                self._bits_of_compressed_grad = lambda x: (2*math.ceil(x*(1-self._drop_ratio))+1)*32 # in float
                if 'drop_ratio' in kwargs.keys():
                    self._drop_ratio = kwargs['drop_ratio']
                if 'sample_rate' in kwargs.keys():
                    self._sample_rate = kwargs['sample_rate']
                self._comp_parameters_map['graddrop']['drop_ratio'] = self._drop_ratio
                self._comp_parameters_map['graddrop']['sample_rate'] = self._sample_rate
                self._param_tmp = self._drop_ratio
                self._compression_rate = lambda x: x*(1-self._drop_ratio)
            elif self._comp_alg == 'topkadam':
                self._s_percent = kwargs['s_percent'] if 's_percent' in kwargs.keys() else 0.001
                self._sample_rate = kwargs['sample_rate'] if 'sample_rate' in kwargs.keys() else 0.001
                self.engine = GradientCompressEngine(
                    compressor=TopKAdamCompressor(
                        optimizer=self._optimizer,
                        s_percent=self._s_percent,
                        sample_rate=self._sample_rate)
                )
            elif self._comp_alg == 'adacomp':
                self._compress = mx.nd.contrib.ada_comp
                self._decompress = mx.nd.contrib.ada_comp_r
                self._residual = True
                self._T = 500
                self._bits_of_compressed_grad = lambda x: x
                if 'T' in kwargs.keys():
                    self._T = kwargs['T']
                self._comp_parameters_map['adacomp']['T'] = self._T
                self._compression_rate = 1/self._T
            elif self._comp_alg == 'powersgd':
                self._compress = None
                self._decompress = None
                self._residual = True
                self._decomp_rank = kwargs['rank'] if 'rank' in kwargs.keys() else 4
                self.engine = GradientCompressEngine(
                    compressor=PowerSGDCompressor(decomp_rank=self._decomp_rank)
                )
            else:
                warnings.warn("The compression algorith %s is not supported right now"%(self._comp_alg))
    
    def binary(self, num):
        return ''.join(bin(c).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', num))

    def float32_to_int32(self, num):
        b = self.binary(num)
        if b[0]=='0':
            return int(b,2)
        else:
            return -int(b[1:],2)

    def __getattr__(self, item):
        return getattr(self._optimizer, item)

    def create_state_multi_precision(self, index, weight):
        return self._optimizer.create_state_multi_precision(index, weight)


    def get_size_in_header(self, A):
        a = A[0] + A[1]*256 + A[2]*65536 + A[3]*65536*256
        b = A[3] + A[2]*256 + A[1]*65536 + A[0]*65536*256
        # print("a={}\tb={}".format(a,b))
        return a

    def initialize(self, index, grad):
        if index in self._comp_gpus: # initialized
            return
        interval = grad.size
        self._comp_gpus[index] = mx.nd.zeros(shape=interval*5, dtype='uint8', ctx=grad.context)
        self._comp_cpus[index] = mx.nd.zeros(shape=interval*8, dtype='uint8', ctx=mx.context.cpu_pinned())
        self._resd_gpus[index] = mx.nd.zeros(shape=interval, dtype=grad.dtype, ctx=grad.context)

    def partition_initialize(self, index, grad):
        if index in self._comp_gpus: # initialized
            return
        self._comp_gpus[index] = list()
        self._comp_cpus[index] = list()
        self._resd_gpus[index] = list()
        interval = grad.size // size()
        for i in range(size()):
            if i == size() - 1:
                interval += grad.size%size()
            self._comp_gpus[index].append(mx.nd.zeros(shape=interval*5,
                                                      dtype='uint8',
                                                      ctx=grad.context))
            self._comp_cpus[index].append(mx.nd.zeros(shape=interval*8,
                                                      dtype='uint8',
                                                      ctx=mx.context.cpu_pinned()))
            self._resd_gpus[index].append(mx.nd.zeros(shape=interval,
                                                      dtype=grad.dtype,
                                                      ctx=grad.context))

    def _do_partition_allreduce(self, i, index, grad, batchid=0):
        self.partition_initialize(index[i], grad[i])
        # partition and do compression and gather
        interval = grad[i].size // size()
        for part_root in range(size()):
            start = part_root * interval
            end = (part_root+1) * interval
            if part_root == size()-1:
                end = grad[i].size
                interval += grad[i].size%size()
            compressed_size = int((self._bits_of_compressed_grad(interval)+7)/8) # ceil in uint8
            if rank() != part_root: # do compression
                self._comp_parameters_map[self._comp_alg]['data'] = grad[i].reshape(grad[i].size)[start:end]
                self._comp_parameters_map[self._comp_alg]['out'] = self._comp_gpus[index[i]][part_root]
                if self._residual:
                    self._comp_parameters_map[self._comp_alg]['residual'] = self._resd_gpus[index[i]][part_root]
                self._compress(**self._comp_parameters_map[self._comp_alg])
                self._comp_gpus[index[i]][part_root][:compressed_size].copyto(
                    self._comp_cpus[index[i]][part_root][:compressed_size])
            gather_(self._comp_cpus[index[i]][part_root], self._comp_cpus[index[i]][part_root],
                    grad[i], part_root, num_elem=compressed_size, name=str(index[i])+"-"+str(part_root), batchid=batchid)

        # partition and do decompression at root and bcast
        interval = grad[i].size // size()
        for part_root in range(size()):
            start = part_root * interval
            end = (part_root+1) * interval
            if part_root == size()-1:
                end = grad[i].size
                interval += grad[i].size%size()
            compressed_size = int((self._bits_of_compressed_grad(interval)+7)/8) # ceil in uint8
            if rank() == part_root:
                tail = interval%4; tail = 0 if tail == 0 else 4 - tail
                for idx in range(size()):
                    if idx != part_root:
                        begin = idx * compressed_size; stop = begin + compressed_size
                        self._comp_cpus[index[i]][part_root][begin:stop].copyto(
                            self._comp_gpus[index[i]][part_root][:compressed_size])
                        self._decomp_parameters_map[self._comp_alg]['data'] = self._comp_gpus[index[i]][part_root][:compressed_size]
                        self._decomp_parameters_map[self._comp_alg]['is_add_to'] = 1
                        self._decomp_parameters_map[self._comp_alg]['out'] = grad[i].reshape(grad[i].size)[start:end]
                        self._decomp_parameters_map['ecq']['is_add_to'] = 0
                        self._decomp_parameters_map['terngrad']['tail'] = tail
                        self._decompress(**self._decomp_parameters_map[self._comp_alg])
                self._comp_parameters_map[self._comp_alg]['data'] = grad[i].reshape(grad[i].size)[start:end]
                self._comp_parameters_map[self._comp_alg]['out'] = self._comp_gpus[index[i]][part_root]
                if self._residual:
                    self._comp_parameters_map[self._comp_alg]['residual'] = self._resd_gpus[index[i]][part_root]
                self._compress(**self._comp_parameters_map[self._comp_alg])
                self._comp_gpus[index[i]][part_root][:compressed_size].copyto(
                    self._comp_cpus[index[i]][part_root][:compressed_size])
            mpibroadcast_(self._comp_cpus[index[i]][part_root], part_root,
                            num_elem=compressed_size, name=str(index[i])+'-'+str(part_root),
                            batchid=batchid)

        # partition and do decompression at non root
        interval = grad[i].size // size()
        for part_root in range(size()):
            start = part_root * interval
            end = (part_root+1) * interval
            if part_root == size()-1:
                end = grad[i].size
                interval += grad[i].size%size()
            compressed_size = int((self._bits_of_compressed_grad(interval)+7)/8) # ceil in uint8
            if rank() != part_root:
                self._comp_cpus[index[i]][part_root][:compressed_size].copyto(
                    self._comp_gpus[index[i]][part_root][:compressed_size])
                self._decomp_parameters_map[self._comp_alg]['data'] = self._comp_gpus[index[i]][part_root][:compressed_size]
                self._decomp_parameters_map[self._comp_alg]['is_add_to'] = 0
                self._decomp_parameters_map[self._comp_alg]['out'] = grad[i].reshape(grad[i].size)[start:end]
                tail = interval%4; tail = 0 if tail == 0 else 4 - tail
                self._decomp_parameters_map['terngrad']['tail'] = tail
                self._decompress(**self._decomp_parameters_map[self._comp_alg])
        grad[i].__idiv__(size())

    def _do_allreduce(self, i, index, grad, batchid=0):
        self.initialize(index[i], grad[i])
        if self._sparsity:
            assert self._compression_rate(size()) < 0.5, "compression rate too large! may use quantization."
        assert self._bits_of_compressed_grad(grad[i].size) <= 8*self._comp_cpus[index[i]].size, "need more space for gathering!"

        index = list(index)
        # index[i] = self._key_start
        # self._key_start += 1
        rootid = index[i]%size()
        # rootid = 0
        compressed_size = int((self._bits_of_compressed_grad(grad[i].size)+7)/8) # ceil in uint8
        # compressed_size can be extracted from compressed data
        if rank() != rootid:
            self._comp_parameters_map[self._comp_alg]['data'] = grad[i].reshape(grad[i].size)
            self._comp_parameters_map[self._comp_alg]['out'] = self._comp_gpus[index[i]]
            if self._residual:
                self._comp_parameters_map[self._comp_alg]['residual'] = self._resd_gpus[index[i]]
            self._compress(**self._comp_parameters_map[self._comp_alg])
            self._comp_gpus[index[i]][:compressed_size].copyto(self._comp_cpus[index[i]][:compressed_size])
            # print(socket.gethostname(), comp_cpu[i][0:4])
        gather_(self._comp_cpus[index[i]], self._comp_cpus[index[i]], grad[i], rootid, 
                num_elem=compressed_size, name=str(index[i]), batchid=batchid)

        if self._sparsity:
            compressed_size_new = self._bits_of_compressed_grad(grad[i].size*size())/8 # in uint8
        else:
            compressed_size_new = compressed_size
        compressed_size_new = int(compressed_size_new)

        if rank() == rootid:
            tail = grad[i].size%4; tail = 0 if tail==0 else 4-tail # for terngrad
            for idx in range(size()):
                if idx != rootid:
                    start = idx*compressed_size; offset = start+compressed_size
                    self._comp_cpus[index[i]][start:offset].copyto(self._comp_gpus[index[i]][:compressed_size])
                    self._decomp_parameters_map[self._comp_alg]['data'] = self._comp_gpus[index[i]][:compressed_size]
                    self._decomp_parameters_map[self._comp_alg]['is_add_to'] = 1
                    self._decomp_parameters_map[self._comp_alg]['out'] = grad[i].reshape(grad[i].size)
                    self._decomp_parameters_map['ecq']['is_add_to'] = 0
                    self._decomp_parameters_map['terngrad']['tail'] = tail # only for terngrad
                    self._decompress(**self._decomp_parameters_map[self._comp_alg])

            if self._sparsity:
                if self._comp_alg == 'dgc':
                    self._comp_parameters_map['dgc']['s_percent'] *= size()
                elif self._comp_alg == 'graddrop':
                    self._comp_parameters_map['graddrop']['drop_ratio'] \
                        -= (1-self._comp_parameters_map['graddrop']['drop_ratio'])*(size()-1)

            self._comp_parameters_map[self._comp_alg]['data'] = grad[i].reshape(grad[i].size)
            self._comp_parameters_map[self._comp_alg]['out'] = self._comp_gpus[index[i]]
            if self._residual:
                self._comp_parameters_map[self._comp_alg]['residual'] = self._resd_gpus[index[i]]
            self._compress(**self._comp_parameters_map[self._comp_alg])
            self._comp_gpus[index[i]][:compressed_size_new].copyto(self._comp_cpus[index[i]][:compressed_size_new])

            if self._sparsity and self._compression_rate(size()) < 0.5:
                if self._comp_alg == 'dgc':
                    self._comp_parameters_map['dgc']['s_percent'] = self._param_tmp
                elif self._comp_alg == 'graddrop':
                    self._comp_parameters_map['graddrop']['drop_ratio'] = self._param_tmp
        mpibroadcast_(self._comp_cpus[index[i]], rootid, num_elem=compressed_size_new, name=str(index[i]), batchid=batchid)

        if rank() != rootid:
            self._comp_cpus[index[i]][:compressed_size_new].copyto(self._comp_gpus[index[i]][:compressed_size_new])
            self._decomp_parameters_map[self._comp_alg]['data'] = self._comp_gpus[index[i]][:compressed_size_new]
            self._decomp_parameters_map[self._comp_alg]['is_add_to'] = 0
            self._decomp_parameters_map[self._comp_alg]['out'] = grad[i].reshape(grad[i].size)
            tail = grad[i].size % 4; tail = 0 if tail == 0 else 4-tail # only for terngrad
            self._decomp_parameters_map['terngrad']['tail'] = tail

            self._decompress(**self._decomp_parameters_map[self._comp_alg])
        grad[i].__idiv__(size()) # average

    def update(self, index, weight, grad, state):
        raise NotImplementedError()

    def update_multi_precision(self, index, weight, grad, state):
        self.do_allreduce(index, grad, state)
        if self._optimizer_compression:
            if not isinstance(index, (list, tuple)):
                index = [index]
                weight = [weight]
                grad = [grad]
                state = [state]
            for i in range(len(index)):
                if grad[i].size >= self._threshold:
                    mx.nd.sgd_update(weight[i], grad[i], out=weight[i], lr=-1.0, wd=0.0)
                else:
                    self._optimizer.update_multi_precision(index[i], weight[i], grad[i], state[i])
        else:
            self._optimizer.update_multi_precision(index, weight, grad, state)

    def do_allreduce(self, index, grad, state):
        if not isinstance(index, (tuple, list)): # Bert or other nlp models
            index = [index]
            grad = [grad]
        for i in range(len(index)):
            if self._do_compression and grad[i].size >= self._threshold:
                if self._quantization and grad[i].size*4 >= self._partition_byte:
                    self._do_partition_allreduce(i, index, grad, batchid=batchid)
                elif self.engine is not None:
                    self.engine.do_compressed_communication(index[i], grad[i], state[i])
                else:
                    self._do_allreduce(i, index, grad, batchid=batchid)
            else:
                allreduce_(grad[i], average=True, name=str(index[i]), batchid=0)

    def set_learning_rate(self, lr):
        self._optimizer.set_learning_rate(lr)

    def set_lr_mult(self, args_lr_mult):
        self._optimizer.set_lr_mult(args_lr_mult)

    def set_wd_mult(self, args_wd_mult):
        self._optimizer.set_wd_mult(args_wd_mult)


# DistributedTrainer, a subclass of MXNet gluon.Trainer.
# There are two differences between DistributedTrainer and Trainer:
# 1. DistributedTrainer calculates gradients using Horovod allreduce
#    API while Trainer does it using kvstore push/pull APIs;
# 2. DistributedTrainer performs allreduce(summation) and average
#    while Trainer only performs allreduce(summation).
class DistributedTrainer(mx.gluon.Trainer):
    def __init__(self, params, optimizer, optimizer_params=None):
        if isinstance(optimizer, DistributedOptimizer):
            optimizer = optimizer._optimizer
            warnings.warn("DistributedTrainer does not take DistributedOptimizer "
                          "as its optimizer. We have unwrapped it for you.")

        super(DistributedTrainer, self).__init__(
            params, optimizer, optimizer_params=optimizer_params, kvstore=None)

    def _allreduce_grads(self):
        for i, param in enumerate(self._params):
            if param.grad_req != 'null':
                allreduce_(param.list_grad()[0], average=True, name=str(i))


# Wrapper to inject Horovod broadcast after parameter initialization
def _append_broadcast_init(param, root_rank):
    init_impl = getattr(param, '_init_impl')
    def wrapped_init_impl(self, *args, **kwargs):
        init_impl(*args, **kwargs)
        broadcast_(self.data(), root_rank=root_rank)
        self.data().wait_to_read()
    return wrapped_init_impl


def broadcast_parameters(params, root_rank=0):
    """
    Broadcasts the parameters from root rank to all other processes.
    Typical usage is to broadcast the `Module.get_params()` or the
    `Block.collect_params()`.

    Arguments:
        params: One of the following:
            - dict of parameters to broadcast
            - ParameterDict to broadcast
        root_rank: The rank of the process from which parameters will be
                   broadcasted to all other processes.
    """
    tensors = []
    if isinstance(params, list):
        for p in params:
            try:
                tensors.append(p.data())
            except mx.gluon.parameter.DeferredInitializationError:
                new_init = _append_broadcast_init(p, root_rank)
                p._init_impl = types.MethodType(new_init, p)
    elif isinstance(params, dict):
        tensors = [p for _, p in sorted(params.items())]
    elif isinstance(params, mx.gluon.parameter.ParameterDict):
        for _, p in sorted(params.items()):
            try:
                tensors.append(p.data())
            except mx.gluon.parameter.DeferredInitializationError:
                # Inject wrapper method with post-initialization broadcast to
                # handle parameters with deferred initialization
                new_init = _append_broadcast_init(p, root_rank)
                p._init_impl = types.MethodType(new_init, p)
    else:
        raise ValueError('invalid params of type: %s' % type(params))

    # Run broadcasts.
    for i, tensor in enumerate(tensors):
        broadcast_(tensor, root_rank, name=str(i))

    # Make sure tensors pushed to MXNet engine get processed such that all
    # workers are synced before starting training.
    for tensor in tensors:
        tensor.wait_to_read()
