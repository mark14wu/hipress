# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications copyright (C) 2017 Uber Technologies, Inc.
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
# pylint: disable=g-short-docstring-punctuation
"""## Communicating Between Processes with MPI

TensorFlow natively provides inter-device communication through send and
receive ops and inter-node communication through Distributed TensorFlow, based
on the same send and receive abstractions. On HPC clusters where Infiniband or
other high-speed node interconnects are available, these can end up being
insufficient for synchronous data-parallel training (without asynchronous
gradient descent). This module implements a variety of MPI ops which can take
advantage of hardware-specific MPI libraries for efficient communication.
"""

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from horovod.common import check_extension
from horovod.tensorflow.compression import Compression
from horovod.tensorflow.mpi_ops import (_allreduce, allgather, broadcast, init,
                                        local_rank, local_size,
                                        mpi_threads_supported, rank, shutdown,
                                        size, gather, cbroadcast, _normalize_name)
from horovod.tensorflow.util import _executing_eagerly

check_extension('horovod.tensorflow', 'HOROVOD_WITH_TENSORFLOW', __file__, 'mpi_lib')

import sys
import colorful as cf

import logging
logging.basicConfig(format="%(filename)s[line:%(lineno)d]%(funcName)s\t%(message)s",level=logging.DEBUG)

logger = logging.getLogger(name="check dgc")
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter=logging.Formatter("%(filename)s[line:%(lineno)d]%(funcName)s\t%(message)s")
console.setFormatter(formatter)
logger.addHandler(console)

def broadcast_global_variables(root_rank):
    """Broadcasts all global variables from root rank to all other processes.

    Arguments:
        root_rank: rank of the process from which global variables will be broadcasted
        to all other processes.
    """
    return broadcast_variables(tf.global_variables(), root_rank)


def broadcast_variables(variables, root_rank):
    """Broadcasts variables from root rank to all other processes.

    Arguments:
        variables: variables for broadcast
        root_rank: rank of the process from which global variables will be broadcasted
                   to all other processes.
    """
    return tf.group(*[tf.assign(var, broadcast(var, root_rank))
                      for var in variables])


class BroadcastGlobalVariablesHook(tf.train.SessionRunHook):
    """
    SessionRunHook that will broadcast all global variables from root rank
    to all other processes during initialization.

    This is necessary to ensure consistent initialization of all workers when
    training is started with random weights or restored from a checkpoint.
    """

    def __init__(self, root_rank, device=''):
        """Construct a new BroadcastGlobalVariablesHook that will broadcast all
        global variables from root rank to all other processes during initialization.

        Args:
          root_rank:
            Rank that will send data, other ranks will receive data.
          device:
            Device to be used for broadcasting. Uses GPU by default
            if Horovod was build with HOROVOD_GPU_BROADCAST.
        """
        super(BroadcastGlobalVariablesHook, self).__init__()
        self.root_rank = root_rank
        self.bcast_op = None
        self.device = device

    def begin(self):
        if not self.bcast_op or self.bcast_op.graph != tf.get_default_graph():
            with tf.device(self.device):
                self.bcast_op = broadcast_global_variables(self.root_rank)

    def after_create_session(self, session, coord):
        session.run(self.bcast_op)

class Gather_CBroadcast():
    def _terngrad_init(self):
        algorithm_name = 'terngrad'
        self._param[algorithm_name] = {
            'bitwidth':2,
            'random':0
        }
        terngrad_module = tf.load_op_library('./terngrad.so')
        self._terngrad = terngrad_module.terngrad
        self._terngrad_r = terngrad_module.terngrad_r
    def _dgc_init(self):
        algorithm_name = 'dgc'
        self._param[algorithm_name] = {
            'sample_rate': 0.001,
            's_percent': 0.001
        }
        dgc_module = tf.load_op_library('./dgc.so')
        self._dgc = dgc_module.dgc
        self._dgc_r = dgc_module.dgc_r

    def root_rank(self):
        return 0
    def is_root(self):
        return rank()==self.root_rank()
    def __init__(self, threshold=262144):
        self._param=dict()
        self._terngrad_init()
        self._dgc_init()
        self._threshold = threshold
        self._space=dict()
        self._size=dict()
        self._name_root={"index":0}
    
    def get_size(self, shape):
        ans=1
        for v in shape:
            ans*=v
        return ans

    def allreduce(self, tensor, average=True, device_dense='', device_sparse='',
              compression=Compression.none):
        if isinstance(tensor, tf.IndexedSlices):
            with tf.device(device_sparse):
                # For IndexedSlices, do two allgathers instead of an allreduce.
                horovod_size = tf.cast(size(), tensor.values.dtype)
                values = allgather(tensor.values)
                indices = allgather(tensor.indices)

                # To make this operation into an average, divide allgathered values by
                # the Horovod size.
                new_values = tf.div(values, horovod_size) if average else values
            return tf.IndexedSlices(new_values, indices,
                                    dense_shape=tensor.dense_shape)
        else:
            with tf.device(device_dense):
                horovod_size = tf.cast(size(), dtype=tensor.dtype)
                tensor_compressed, ctx = compression.compress(tensor)
                summed_tensor_compressed = _allreduce(tensor_compressed)
                summed_tensor = compression.decompress(summed_tensor_compressed, ctx)
                new_tensor = (tf.div(summed_tensor, horovod_size)
                            if average else summed_tensor)
            return new_tensor

    def run(self, tensor, average=True, device_dense='', device_sparse=''):
        size=self.get_size(tensor.shape)
        if (size < 262144):
            # if (size < self._threshold):
            logging.info(cf.green("call allreduce, name={}, shape={}, size={}".format(tensor.name, tensor.shape, size)))
            return self.allreduce(tensor, average=average, device_dense=device_dense, device_sparse=device_sparse)
        # elif size < 8388608:
        elif size < 9999999999999:
            logging.info(cf.green("call gather_broadcast, name={}, shape={}, size={}".format(tensor.name, tensor.shape, size)))
            return self.gather_broadcast(tensor,average=average,device_dense=device_dense, device_sparse=device_sparse)
        else:
            logging.info(cf.green("call gather_broadcast_load_balance, name={}, shape={}, size={}".format(tensor.name, tensor.shape, size)))
            return self.gather_broadcast_load_balance(tensor,average=average,device_dense=device_dense, device_sparse=device_sparse)

    def gather_broadcast_load_balance(self, tensor, average=True, device_dense='', device_sparse=''):
        if isinstance(tensor, tf.IndexedSlices):
            with tf.device(device_sparse):
                horovod_size = tf.cast(size(), tensor.values.dtype)
                values = allgather(tensor.valuese)
                indices = allgather(tensor.indices)
                new_values = tf.div(values, horovod_size) if average else values
            return tf.IndexedSlices(new_values, indices, dense_shape=tensor.dense_shape)
        else:
            with tf.device(device_dense):
                horovod_size = tf.cast(size(), dtype=tensor.dtype)
                tensor_size = self.get_size(tensor.shape)
                tensor_name = tensor.name
                perN = (tensor_size+size()-1)//size()
                lastN = tensor_size - perN * (size()-1)
                currNs = [lastN if i == size()-1 else perN for i in range(size())]
                heads = [perN*i for i in range(size())]
                Qs=dict()
                gather_results=dict()
                Q_sizes=dict()
                for par in range(size()):
                    Qs[par] = self._dgc(
                        tensor,
                        start=heads[par],
                        len=currNs[par],
                        s_percent = self._param['dgc']['s_percent'],
                        sample_rate = self._param['dgc']['sample_rate'],
                        bypass=0
                    )
                    Q_sizes[par] = tf.size(Qs[par])
                    logging.info(cf.red("par={}:\thead={}\tcurrN={}".format(par, heads[par], currNs[par])))
                for par in range(size()):
                    gather_results[par]=gather(
                        Qs[par],
                        root_rank = par,
                        batch_id = 0,
                        name = 'HorovodGather_{}-{}'.format(_normalize_name(tensor_name), par)
                    )
                for par in range(size()):
                    if rank() == par:
                        for i in range(size()):
                            if i != par:
                                tensor = self._dgc_r(
                                    gather_results[par][Q_sizes[par]*i: Q_sizes[par]*(i+1)],
                                    tensor,
                                    is_add_to = 1,
                                    offset=heads[par],
                                    bypass=0,
                                    tag=233
                                )
                        Qs[par] = self._dgc(
                            tensor,
                            start=heads[par],
                            len=currNs[par],
                            s_percent=self._param['dgc']['s_percent'],
                            sample_rate=self._param['dgc']['sample_rate'],
                            bypass=0
                        )
                    else:
                        Qs[par] = gather_results[par][Q_sizes[par]:Q_sizes[par]*2]
                        # pass
                # for par in range(size()):
                #     if rank()==par:
                #         for i in range(size()):
                #             if i!=par:
                #                 logger.info(cf.yellow("calling dgc_r"))
                #                 tensor = self._dgc_r(
                #                     gather_results[par][Q_sizes[par]*i: Q_sizes[par]*i+Q_sizes[par]],
                #                     tensor, 
                #                     is_add_to = 1,
                #                     offset=heads[par],
                #                     bypass=0,
                #                     tag=240
                #                 )
                #         Qs[par] = self._dgc(
                #             tensor,
                #             start=heads[par],
                #             len=currNs[par],
                #             s_percent = self._param['dgc']['s_percent'],
                #             sample_rate = self._param['dgc']['sample_rate'],
                #             bypass=0
                #         )
                #     else:
                #         Qs[par] = gather_results[par][:Q_sizes[par]]

                for par in range(size()):
                    Qs[par] = cbroadcast(
                        Qs[par],
                        root_rank = par,
                        batch_id=0,
                        name='HorovodCBroadcast_{}-{}'.format(_normalize_name(tensor_name), par)
                    )
                for par in range(size()):
                    logging.fatal(cf.yellow("calling dgc_r"))
                    tensor = self._dgc_r(
                        Qs[par],
                        tensor,
                        is_add_to=0,
                        offset=heads[par],
                        bypass=0,
                        tag=289
                    )                
                # for par in range(size()):
                #     Qs[par] = cbroadcast(
                #         Qs[par],
                #         root_rank = par,
                #         batch_id=0,
                #         name='HorovodCBroadcast_{}-{}'.format(_normalize_name(tensor_name), par)
                #     )
                #     tensor = self._dgc_r(
                #         Qs[par],
                #         tensor,
                #         is_add_to=0,
                #         offset=heads[par],
                #         bypass=0,
                #         tag=289
                #     )                
            return tensor

    def get_size(self, shape):
        ans = 1
        for v in shape:
            ans*=v
        return int(ans)

    def extract_addN(self, s:str):
        s:str = _normalize_name(s)
        p = s.rfind('_')
        q = s.rfind('_',end=p-1)
        print(f"extract:\t{s} --> {s[q:p]}")
        return s[q:p]


    def gather_broadcast(self, tensor, average=True, device_dense='', device_sparse=''):
        # logging.info(cf.green("call gather_broadcast, name={}, shape={}".format(tensor.name, tensor.shape)))
        # def get_size(shape):
        #     '''
        #     assume shape is knownn
        #     '''
        #     ans = 1
        #     for v in shape:
        #         ans*=v
        #     return ans
        # logging.info(cf.green("size={}".format(get_size(tensor.shape))))

        if isinstance(tensor, tf.IndexedSlices):
            with tf.device(device_sparse):
                horovod_size = tf.cast(size(), tensor.values.dtype)
                values = allgather(tensor.valuese)
                indices = allgather(tensor.indices)
                new_values = tf.div(values, horovod_size) if average else values
            return tf.IndexedSlices(new_values, indices, dense_shape=tensor.dense_shape)
        else:
            with tf.device(device_dense):
                # tensor_size = tf.size(tensor)
                tensor_size = self.get_size(tensor.shape)
                import math
                print(cf.red(f"type(tensor_size)={type(tensor_size)}"))
                Q_size = (math.ceil(tensor_size*self._param['dgc']['s_percent'])*2)*4+4
                print(cf.red(f"tensor.shape={tensor.shape}, size={self.get_size(tensor.shape)}, Q_size={Q_size}"))
                tensor_name = tensor.name
                if tensor_name not in self._name_root:
                    self._name_root[tensor_name] = self._name_root['index']
                    self._name_root['index'] = (self._name_root['index']+1) % size()
                my_root = self._name_root[tensor_name]
                my_root = 0

                if rank()==my_root:
                    # Q = s
                    Q = self._dgc(
                        tensor,
                        start=0,
                        len=-1,
                        s_percent = self._param['dgc']['s_percent'],
                        sample_rate = self._param['dgc']['sample_rate'],
                        bypass=0
                    )
                    Q_size = tf.size(Q)
                    gather_result = gather(
                        Q,
                        root_rank = my_root,
                        batch_id=0,
                        name=f'HorovodGather_{_normalize_name(tensor_name)}'
                    )
                    for i in range(size()):
                        if i != my_root:
                            tensor - self._dgc_r(
                                gather_result[Q_size*i:Q_size*(i+1)],
                                tensor,
                                is_add_to = 1,
                                offset= 0,
                                bypass = 0,
                                tag = 378
                            )
                    Q = self._dgc(
                        tensor,
                        start=0,
                        len=-1,
                        s_percent=self._param['dgc']['s_percent'],
                        sample_rate = self._param['dgc']['sample_rate'],
                        bypass = 0,
                    )
                    # Q_size = tf.size(Q)
                    # gather_result = gather(
                    #     Q,
                    #     root_rank = my_root,
                    #     batch_id=0,
                    #     name='HorovodGather_{}'.format(_normalize_name(tensor_name))
                    # )
                    # for i in range(size()):
                    #     if i != my_root:
                    #         tensor = self._dgc_r(
                    #             gather_result[Q_size*i:Q_size*(i+1)],
                    #             # tf.slice(gather_result,[Q_size*i], [Q_size]),
                    #             tensor,
                    #             is_add_to = 1,
                    #             offset=0,
                    #             bypass=0,
                    #             tag=370
                    #         )
                    # Q = self._dgc(
                    #     tensor,
                    #     start=0,
                    #     len=-1,
                    #     s_percent=self._param['dgc']['s_percent'],
                    #     sample_rate=self._param['dgc']['sample_rate'],
                    #     bypass=0
                    # )
                    Q = cbroadcast(
                        Q,
                        root_rank = my_root,
                        batch_id=0,
                        name='HorovodCBroadcast_{}'.format(_normalize_name(tensor_name))
                    )
                    # Q = broadcast(
                    #     Q,
                    #     root_rank = my_root,
                    #     name='HorovodBroadcast_{}'.format(_normalize_name(tensor_name))
                    # )
                    # logger.info(cf.yellow("calling dgc_r"))
                    tensor = self._dgc_r(
                        Q,
                        tensor,
                        is_add_to = 0, 
                        offset=0,
                        bypass=0,
                        tag=400
                    )
                else:
                    Q = self._dgc(
                        tensor,
                        start=0,
                        len=-1,
                        s_percent = self._param['dgc']['s_percent'],
                        sample_rate = self._param['dgc']['sample_rate'],
                        bypass=0
                    )
                    Q_size = tf.size(Q)
                    gather_result = gather(
                        Q,
                        root_rank = my_root,
                        batch_id = 0,
                        name = f'HorovodGather_{_normalize_name(tensor_name)}'
                    )
                    # Q_size = tf.size(Q)
                    # gather_result = gather(
                    #     Q,
                    #     root_rank = my_root,
                    #     batch_id=0,
                    #     name='HorovodGather_{}'.format(_normalize_name(tensor_name))
                    # )
                    # Q=gather_result[Q_size:Q_size*2]
                    Q=cbroadcast(
                        Q,
                        root_rank = my_root,
                        batch_id=0,
                        name='HorovodCBroadcast_{}'.format(_normalize_name(tensor_name))
                    )
                    # Q = broadcast(
                    #     Q,
                    #     root_rank = my_root,
                    #     name='HorovodBroadcast_{}'.format(_normalize_name(tensor_name))
                    # )
                    logger.info(cf.yellow("calling dgc_r"))
                    tensor = self._dgc_r(
                        Q,
                        tensor,
                        is_add_to = 0, 
                        offset=0,
                        bypass=0,
                        tag=453
                    )
            return tensor



class DistributedOptimizer(tf.train.Optimizer):
    """An optimizer that wraps another tf.Optimizer, using an allreduce to
    average gradient values before applying gradients to model weights."""
    def __init__(self, optimizer, name=None, use_locking=False, device_dense='',
                 device_sparse='', compression=Compression.none,
                 sparse_as_dense=False):
        """Construct a new DistributedOptimizer, which uses another optimizer
        under the hood for computing single-process gradient values and
        applying gradient updates after the gradient values have been averaged
        across all the Horovod ranks.

        Args:
          optimizer:
            Optimizer to use for computing gradients and applying updates.
          name:
            Optional name prefix for the operations created when applying
            gradients. Defaults to "Distributed" followed by the provided
            optimizer type.
          use_locking:
            Whether to use locking when updating variables.
            See Optimizer.__init__ for more info.
          device_dense:
            Device to be used for dense tensors. Uses GPU by default
            if Horovod was build with HOROVOD_GPU_ALLREDUCE.
          device_sparse:
            Device to be used for sparse tensors. Uses GPU by default
            if Horovod was build with HOROVOD_GPU_ALLGATHER.
          compression:
            Compression algorithm used during allreduce to reduce the amount
            of data sent during the each parameter update step.  Defaults to
            not using compression.
          sparse_as_dense:
            Treat all sparse gradients as dense tensors.  This can help improve
            performance and memory utilization if the original sparse gradient
            has high density.  Defaults to false.
        """
        if name is None:
            name = "Distributed{}".format(type(optimizer).__name__)
        
        self.gather_cbroadcast = Gather_CBroadcast()
        self._optimizer = optimizer
        self._device_dense = device_dense
        self._device_sparse = device_sparse
        self._compression = compression
        self._sparse_as_dense = sparse_as_dense

        def gather_broadcast_grads(grads):
            with tf.name_scope(self._name + "_GatherBroadcast"):
                if self._sparse_as_dense:
                    grads = [tf.convert_to_tensor(grad)
                            if grad is not None and isinstance(grad, tf.IndexedSlices) else grad
                            for grad in grads]
                # return [self.gather_broadcast(
                return [self.gather_cbroadcast.run(
                        grad,
                        device_dense=self._device_dense,
                        device_sparse = self._device_sparse,
                        )
                        if grad is not None else grad
                        for grad in grads
                        ]

        if _executing_eagerly():
            # self._allreduce_grads = tf.contrib.eager.defun(allreduce_grads)
            self._gather_broadcast_grads = tf.contrib.eager.defun(gather_broadcast_grads)
        else:
            # self._allreduce_grads = allreduce_grads
            self._gather_broadcast_grads = gather_broadcast_grads

        super(DistributedOptimizer, self).__init__(
            name=name, use_locking=use_locking)

    # def compute_gradients(self, *args, **kwargs):
    #     """Compute gradients of all trainable variables.

    #     See Optimizer.compute_gradients() for more info.

    #     In DistributedOptimizer, compute_gradients() is overriden to also
    #     allreduce the gradients before returning them.
    #     """
    #     gradients = self._optimizer.compute_gradients(*args, **kwargs)
    #     if size() > 1:
    #         print("ZQDEBUG:",sys._getframe().f_lineno, "compute_gradients")
    #         grads, vars = zip(*gradients)
    #         avg_grads = self._allreduce_grads(grads)
    #         return list(zip(avg_grads, vars))
    #     else:
    #         return gradients
    def compute_gradients(self, *args, **kwargs):
        gradients = self._optimizer.compute_gradients(*args, **kwargs)
        if size() > 1:
            grads, vars = zip(*gradients)
            avg_grads = self._gather_broadcast_grads(grads)
            return list(zip(avg_grads, vars))
        else:
            return gradients


    def apply_gradients(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.apply_gradients(*args, **kwargs)

    def get_slot(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.get_slot(*args, **kwargs)

    def get_slot_names(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.get_slot_names(*args, **kwargs)

    def variables(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.variables(*args, **kwargs)


if hasattr(tf, 'GradientTape'):
    import colorful as cf
    logging.info(cf.yellow("Warning: function in _DistributedGradientTape has not been replaced"))
    class _DistributedGradientTape(tf.GradientTape):

        def __init__(self, tape, device_dense, device_sparse,
                     compression, sparse_as_dense, persistent=False, watch_accessed_variables=True):
            if hasattr(tape, '_watch_accessed_variables'):
                super(self.__class__, self).__init__(persistent, watch_accessed_variables)
            else:
                super(self.__class__, self).__init__(persistent)
            self._tape = tape
            self._persistent = persistent
            self._watch_accessed_variables = watch_accessed_variables
            self._name = "Distributed"
            self._device_dense = device_dense
            self._device_sparse = device_sparse
            self._compression = compression
            self._sparse_as_dense = sparse_as_dense
            self.gather_cbroadcast = Gather_CBroadcast()

            def allreduce_grads(grads):
                with tf.name_scope(self._name + "_Allreduce"):
                    if self._sparse_as_dense:
                        grads = [tf.convert_to_tensor(grad)
                                 if grad is not None and isinstance(grad, tf.IndexedSlices)
                                 else grad for grad in grads]
                    return [self.gather_cbroadcast.gather_broadcast(
                        grad,
                        device_dense = self._device_dense,
                        device_sparse = self._device_sparse
                    )
                    if grad is not None else grad
                    for grad in grads
                    ]

            self._allreduce_grads = tf.contrib.eager.defun(allreduce_grads)

        def gradient(self, target, sources, output_gradients=None):
            gradients = super(self.__class__, self).gradient(target, sources, output_gradients)
            if size() > 1:
                logging.fatal(tf.yellow("Call allreduce in _DistributedGradientTape"))
                avg_grads = self._allreduce_grads(gradients)
                return avg_grads
            else:
                return gradients


def DistributedGradientTape(gradtape, device_dense='', device_sparse='',
                            compression=Compression.none, sparse_as_dense=False):
    """An tape that wraps another tf.GradientTape, using an allreduce to
    average gradient values before applying gradients to model weights.

    Args:
      gradtape:
        GradientTape to use for computing gradients and applying updates.
      device_dense:
        Device to be used for dense tensors. Uses GPU by default
        if Horovod was build with HOROVOD_GPU_ALLREDUCE.
      device_sparse:
        Device to be used for sparse tensors. Uses GPU by default
        if Horovod was build with HOROVOD_GPU_ALLGATHER.
      compression:
        Compression algorithm used during allreduce to reduce the amount
        of data sent during the each parameter update step.  Defaults to
        not using compression.
      sparse_as_dense:
        Treat all sparse gradients as dense tensors.  This can help improve
        performance and memory utilization if the original sparse gradient
        has high density.  Defaults to false.
    """
    cls = type(gradtape.__class__.__name__, (gradtape.__class__,),
               dict(_DistributedGradientTape.__dict__))
    if hasattr(gradtape, '_watch_accessed_variables'):
        return cls(gradtape._tape, device_dense, device_sparse,
                   compression, sparse_as_dense,
                   gradtape._persistent, gradtape._watch_accessed_variables)
    else:
        return cls(gradtape._tape, device_dense, device_sparse,
                   compression, sparse_as_dense,
                   gradtape._persistent)
