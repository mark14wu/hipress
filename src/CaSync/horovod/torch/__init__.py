from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from contextlib import contextmanager

from horovod.common import check_extension

try:
    check_extension('horovod.torch', 'HOROVOD_WITH_PYTORCH',
                    __file__, 'mpi_lib_v2')
except:
    check_extension('horovod.torch', 'HOROVOD_WITH_PYTORCH',
                    __file__, 'mpi_lib', '_mpi_lib')

from horovod.torch.compression import Compression
from horovod.torch.mpi_ops import allreduce, allreduce_async, allreduce_, allreduce_async_
from horovod.torch.mpi_ops import allgather, allgather_async
from horovod.torch.mpi_ops import broadcast, broadcast_async, broadcast_, broadcast_async_
from horovod.torch.mpi_ops import gather_async_, cbroadcast_async_, get_all_get_all_finished_keys, gather_synchronize, cbroadcast_synchronize
from horovod.torch.mpi_ops import poll, synchronize
from horovod.torch.mpi_ops import init, shutdown
from horovod.torch.mpi_ops import size, local_size, rank, local_rank
from horovod.torch.mpi_ops import mpi_threads_supported

import torch
from torch._six import queue
import collections
import threading


MP_STATUS_CHECK_INTERVAL = 5.0
TIME_INTERVAL = 1 / 10000

import time
import hp_cuda
import math

def _hp_cuda_submit_task(name, tensor, residual, B, C, is_root, task_id):
    if residual is None:
        hp_cuda.submit_task(name, tensor, B, C, is_root, task_id)
    else:
        hp_cuda.submit_task(name, tensor, residual, B, C, is_root, task_id)       

def _malloc_space_xpu( cpu_size : int, gpu_size : int, cpu_type = torch.uint8, gpu_type = torch.uint8):
    _on_gpu = None
    _on_cpu = None
    if cpu_size > 0:
        _on_cpu = torch.zeros(cpu_size, dtype=cpu_type).pin_memory()
    
    if gpu_size > 0:
        if gpu_type == torch.uint8:
            _on_gpu = torch.cuda.ByteTensor(gpu_size)
        elif gpu_type == torch.float32:
            _on_gpu = torch.cuda.FloatTensor(gpu_size)
        else:
            print("_malloc_space_xpu: unknown gpu_type: {}, only support torch.uint8 or torch.float32".format(gpu_type))
            raise(ValueError)
        
    return _on_cpu, _on_gpu


def _compute_auxiliary_size( size : int, algorithm_name : str, params : dict, hvd_size : int):
    if algorithm_name=='terngrad':
        data_per_byte = 8 / params['bitwidth']
        compressed_size =  10 + (size + data_per_byte - 1) // data_per_byte
        CPU_size = hvd_size * compressed_size
        GPU_size = compressed_size
        return int(compressed_size), int(CPU_size), int(GPU_size)
    elif algorithm_name == 'tbq':
        compressed_size = (size + 3) // 4
        CPU_size = hvd_size * compressed_size
        GPU_size = compressed_size
        return int(compressed_size), int(CPU_size), int(GPU_size)
    
    elif algorithm_name == 'graddrop':
        compressed_size = 4 + int( 2 * math.ceil( size * ( 1 - params['drop_ratio'] ) ) ) * 4
        CPU_size = hvd_size * compressed_size
        GPU_size = size * 4
        return int(compressed_size), int(CPU_size), int(GPU_size)
    
    else:
        print("unknown algorithm_name:{}".format(algorithm_name))
        raise(ValueError)


#input_queue : (tensor, name, None)
#output_queue : (tensor, name, handle)
def allreduce_thread_loop_(input_queue, output_queue, device_id):

    torch.cuda.set_device(device_id)
    torch.set_num_threads(1)
    import time

    _allreduce_handle = dict()
    _allredude_set = dict()
    _allreduce_index = dict()


    last_cycle_time = time.time()
    while True:
        start_time = time.time()
        sleep_time = (last_cycle_time + TIME_INTERVAL - start_time) 
        if sleep_time > 0:
            time.sleep(sleep_time)
        last_cycle_time = time.time()

        while not input_queue.empty():
            r = input_queue.get()
            index, tensor, name = r
            handle = allreduce_async_(tensor, average=True, name=name)
            _allreduce_handle[name] = handle
            _allreduce_index[name] = index
            #output_queue.put((index, tensor, name, None))

        for name in get_all_get_all_finished_keys(30):
            tensor = synchronize(_allreduce_handle[name])
            output_queue.put((_allreduce_index[name], tensor, name, None))

    return True


#input_queue : (tensor, name, root_id)
#output_queue : (tensor, name, None)
def compression_thread_loop_(input_queue, output_queue, algorithm_name, alg_params, hvd_size, hvd_rank, device_id):
    #assert algorithm_name == 'terngrad'
    torch.cuda.set_device(device_id)
    torch.set_num_threads(1)

    _comp_cpu = dict()
    _comp_res = dict()
    _comp_gpu = dict()
    _comp_set = dict()
    _comp_compressed_size = dict()
    _param = dict()
    _gather_handles = dict()
    _cboardcast_handles = dict()
    _root_id = dict()
    _index = dict()
    
    is_need_residual = None
    
    if algorithm_name == 'tbq' or algorithm_name == 'graddrop': 
        is_need_residual = True 
    else: 
        is_need_residual = False
    #algorithm_name = 'tbq'
    #_param[algorithm_name] = {"threshold" : 2}

    #alg_name_, bitwidth, enable_random, device_id, rank_id, size
    #init hp_cuda backthread 
    hp_cuda.init(
        algorithm_name, 
        alg_params, 
        device_id, 
        hvd_rank, 
        hvd_size
    )

    __root = 0
    
    last_cycle_time = time.time()
    while True:
        start_time = time.time()
        sleep_time = (last_cycle_time + TIME_INTERVAL - start_time) 
        if sleep_time > 0:
            time.sleep(sleep_time)
        last_cycle_time = time.time()

        #root_gather_list = []
        while not input_queue.empty():
            r = input_queue.get()
            index, tensor, name = r

            if name not in _comp_set:
                compressed_size, CPU_size, GPU_size = _compute_auxiliary_size(
                    tensor.numel(),
                    algorithm_name, 
                    alg_params,
                    hvd_size
                )
                C, B = _malloc_space_xpu(CPU_size, GPU_size)
                if is_need_residual:
                    _comp_res[name] = torch.zeros(tensor.numel(), device=device_id)
                else:
                    _comp_res[name] = None
                _comp_cpu[name] = C
                _comp_gpu[name] = B
                _comp_compressed_size[name] = compressed_size
                
            
            _comp_set[name] = tensor
            _index[name] = index
            _root_id[name] = __root
            __root = (__root + 1) % hvd_size
                 
            
            #is root
            if _root_id[name] == hvd_rank:
                handle = gather_async_(
                    _comp_cpu[name],
                    _comp_cpu[name],
                    _root_id[name],
                    num_elem = _comp_compressed_size[name],
                    name = name,
                    batchid=0
                )
                _gather_handles[name] = handle
            else:
                # 0 not root
                # 10 comp
                #hp_cuda.submit_task(name, _comp_set[name], _comp_gpu[name], _comp_cpu[name], 0, 10)
                _hp_cuda_submit_task(name, _comp_set[name], _comp_res[name], _comp_gpu[name], _comp_cpu[name], 0, 10)

        
        finished_comp_names = hp_cuda.getResults(10)
        
        #non_root gather
        for name in finished_comp_names:
            handle = gather_async_(
                _comp_cpu[name],
                _comp_cpu[name],
                _root_id[name],
                num_elem = _comp_compressed_size[name],
                name = name,
                batchid=0
            )
            _gather_handles[name] = handle

        #task_id = 10 mean gather task (this is for horovod not for hp_cuda comp)
        finished_gather_names = get_all_get_all_finished_keys(task_id=10)
        
 
        
        for name in finished_gather_names:
            
            _comp_cpu[name] = gather_synchronize(_gather_handles[name])

            #it's root 
            if _root_id[name] == hvd_rank:
                #for root to run  D and C
                # 1 is root
                # 30 is for D and C
                #hp_cuda.submit_task(name, _comp_set[name], _comp_gpu[name], _comp_cpu[name], 1, 30)
                _hp_cuda_submit_task(name, _comp_set[name], _comp_res[name], _comp_gpu[name], _comp_cpu[name], 1, 30)
            else:
                #non_root broadcast
                handle = cbroadcast_async_(
                    _comp_cpu[name],
                    _root_id[name],
                    num_elem = _comp_compressed_size[name],
                    name = name,
                    batchid = 0
                )
                _cboardcast_handles[name] = handle


        finished_root_dc_names = hp_cuda.getResults(30)
        
        #root broadcast
        for name in finished_root_dc_names:
            handle = cbroadcast_async_(
                _comp_cpu[name],
                _root_id[name],
                num_elem = _comp_compressed_size[name],
                name = name,
                batchid = 0
            )
            _cboardcast_handles[name] = handle

        #task_id = 20 mean cbroadcast task
        finished_cbroadcast_names = get_all_get_all_finished_keys(task_id=20)
        
        
        for name in finished_cbroadcast_names:

            _comp_cpu[name] = cbroadcast_synchronize(_cboardcast_handles[name])
            #is root

            if _root_id[name] == hvd_rank:
                #root output
                output_queue.put((_index[name], _comp_set[name], name, None))
            
            else:
                #for non root to D
                #hp_cuda.submit_task(name, _comp_set[name], _comp_gpu[name], _comp_cpu[name], 0, 20)
                _hp_cuda_submit_task(name, _comp_set[name], _comp_res[name], _comp_gpu[name], _comp_cpu[name], 0, 20)

        finished_decomp_names = hp_cuda.getResults(20)


        #non_root finish and output
        for name in finished_decomp_names:
            output_queue.put((_index[name], _comp_set[name], name, None))
        



    hp_cuda.end()

            
            

        






class _DistributedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, named_parameters, compression,
                 backward_passes_per_step=1, **kargs):
        super(self.__class__, self).__init__(params)
        self._compression = compression

        if named_parameters is not None:
            named_parameters = list(named_parameters)
        else:
            named_parameters = []

        # make sure that named_parameters are tuples
        if any([not isinstance(p, tuple) for p in named_parameters]):
            raise ValueError('named_parameters should be a sequence of '
                             'tuples (name, parameter), usually produced by '
                             'model.named_parameters().')

        dups = _DistributedOptimizer.find_duplicates([k for k, _ in named_parameters])
        if len(dups) > 0:
            raise ValueError('Parameter names in named_parameters must be unique. '
                             'Found duplicates: %s' % ', '.join(dups))

        if len(named_parameters) > 0:
            self._parameter_names = {v: k for k, v
                                     in sorted(named_parameters)}
        else:
            self._parameter_names = {v: 'allreduce.noname.%s' % i
                                     for param_group in self.param_groups
                                     for i, v in enumerate(param_group['params'])}

        self._name_parameters = { v : k for k, v in self._parameter_names.items() }
                                     
        self.backward_passes_per_step = backward_passes_per_step
        self._allreduce_delay = {v: self.backward_passes_per_step
                                 for _, v in sorted(named_parameters)}
        
        self._grad_accs = []
        self._requires_update = set()
        self._handles = {}
        self._split_tensor_dict = {}
        self._rev_split_tensor_dict = {}
        self._split_tensor_to_be_finish = {}
        self._compression_threshold = kargs['threshold']
        self._partition_threshold = kargs['partition_threshold']
        
        print(self._compression_threshold, self._partition_threshold)
        
        self.algorithm_name = kargs['algorithm_name']
        self.algorithm_params = kargs['algorithm_params']
        
        if self.algorithm_name == 'tbq':
            if 'threshold' not in self.algorithm_params:
                print("for tbq need parameter threshold")
        elif self.algorithm_name == 'terngrad':
            if 'enable_random' not in self.algorithm_params or 'bitwidth' not in self.algorithm_params:
                print("for terngrad need parameter enable_random and bitwidth")
        elif self.algorithm_name == 'graddrop':
            if 'sample_rate' not in self.algorithm_params or 'drop_ratio' not in self.algorithm_params:
                print("for terngrad need parameter sample_rate and drop_ratio")
        else:
            print("Not support compression algorithm!!! only support TBQ, terngrad and graddrop")
            raise ValueError

        self._size = size()
        self._rank = rank()
        self._device_id = torch.cuda.current_device()
        self._synchronized = False
        self._should_synchronize = True
        print(self._rank)


        if self.__class__.__name__ == 'SGD':
            self._sgd_parameter = { p : (groups['weight_decay'], groups['momentum'], groups['dampening'], groups['nesterov'], groups['lr'])
                for groups in self.param_groups
                for p in groups['params']
            }
    

        if size() > 1:
            self._allreduce_input_queue = queue.Queue()
            self._compression_input_queue = queue.Queue()
            self._output_queue = queue.Queue()

            #allreduce_thread
            self._allredeuce_thread = threading.Thread(
                target=allreduce_thread_loop_,
                args=(self._allreduce_input_queue, self._output_queue, self._device_id)
            )
            self._allredeuce_thread.daemon = True
            self._allredeuce_thread.start()

            #compression_thread
            self._compression_thread = threading.Thread(
                target=compression_thread_loop_,
                args=(
                    self._compression_input_queue, self._output_queue,
                    self.algorithm_name, self.algorithm_params, self._size, 
                    self._rank, self._device_id
                )
            )
            self._compression_thread.daemon = True
            self._compression_thread.start()

                
            #final
            self._register_hooks()
            #wait init
            time.sleep(5)


    def load_state_dict(self, *args, **kwargs):
        self._handles = {}
        self._synchronized = False
        self._should_synchronize = True
        for p in self._allreduce_delay:
            self._allreduce_delay[p] = self.backward_passes_per_step
        super(self.__class__, self).load_state_dict(*args, **kwargs)


    @staticmethod
    def find_duplicates(lst):
        seen = set()
        dups = set()
        for el in lst:
            if el in seen:
                dups.add(el)
            seen.add(el)
        return dups

    
    def _register_hooks(self):
        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    p.grad = p.data.new(p.size()).zero_()
                    self._requires_update.add(p)
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_hook(p))
                    self._grad_accs.append(grad_acc)


    def _submit_task(self, p):
        name = self._parameter_names.get(p)
        tensor = p.grad
        numel = tensor.numel()

        if self._compression_threshold is None or numel < self._compression_threshold:
            r = (-1, tensor, name)
            self._allreduce_input_queue.put(r)
            
        elif numel > self._partition_threshold:

            if name in self._split_tensor_dict or name in self._split_tensor_to_be_finish:
                raise(ValueError)

            number_to_split = self._size

            tensor_after_split = torch.chunk(tensor, number_to_split, dim=0)
            for i in range(number_to_split):
                task_name = name + "+" +  str(i)
                r = (i, tensor_after_split[i], task_name)
                self._compression_input_queue.put(r)
                self._rev_split_tensor_dict[task_name] = name
                
            self._split_tensor_to_be_finish[name] = number_to_split
            self._split_tensor_dict[name] = [None] * number_to_split
        else:
            r = (-1, tensor, name)
            self._compression_input_queue.put(r)

        return True



    def _make_hook(self, p):
        def hook(*ignore):
            if p in self._handles and self._handles[p] is not None:
                if self._allreduce_delay[p] <= 0:
                    raise AssertionError(
                        "Gradients were computed more than "
                        "backward_passes_per_step times before call "
                        "to step(). Increase backward_passes_per_step to "
                        "accumulate gradients locally.")
            assert not p.grad.requires_grad
            assert self._allreduce_delay[p] > 0
            handle = None
            self._allreduce_delay[p] -= 1
            if self._allreduce_delay[p] == 0:
                handle = self._submit_task(p)
                
            self._handles[p] = handle
        return hook


    def synchronize(self):
        missing_p = self._requires_update - set(self._handles.keys())
        for p in missing_p:
            handle = self._submit_task(p)
            print("horovod::torch::_DistributedOptimizer::synchronize::loop1::p,handle,compress_status:",handle,compress_status)
            self._handles[p] = handle

        for p, handle in self._handles.items():
            if handle is None:
                handle = self._submit_task(p)
                print("horovod::torch::_DistributedOptimizer::synchronize::loop2::p,handle,ctx:",handle,compress_status)
                self._handles[p] = handle

        finish_count = 0
        submit_count = len(self._handles)
        while finish_count < submit_count:
            try:
                r = self._output_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue

            index, tensor, name, handle = r
            if index >= 0:
                original_name = self._rev_split_tensor_dict[name]
                self._split_tensor_to_be_finish[original_name] -= 1
                self._split_tensor_dict[original_name][index] = tensor
                if self._split_tensor_to_be_finish[original_name] == 0:
                    tensor = torch.cat(self._split_tensor_dict[original_name], dim=0)
                    name = original_name
                else:
                    continue
            p = self._name_parameters.get(name)
            if handle is not None:
                tensor = synchronize(handle)

            self._allreduce_delay[p] = self.backward_passes_per_step
            p.grad.set_(tensor)
            finish_count += 1
        
        self._handles.clear()
        self._split_tensor_dict.clear()
        self._split_tensor_to_be_finish.clear()
        self._rev_split_tensor_dict.clear()
        
        self._synchronized = True

    @contextmanager
    def skip_synchronize(self):
        self._should_synchronize = False
        try:
            yield
        finally:
            self._should_synchronize = True
            

    def step(self, closure=None):
        if self._should_synchronize:
            if self._synchronized:
                warnings.warn("optimizer.step() called without "
                              "optimizer.skip_synchronize() context after "
                              "optimizer.synchronize(). This can cause training "
                              "slowdown. You may want to consider using "
                              "optimizer.skip_synchronize() context if you use "
                              "optimizer.synchronize() in your code.")
                
            
            if self.__class__.__name__ == 'SGD' and size() > 1:
                missing_p = self._requires_update - set(self._handles.keys())
                for p in missing_p:
                    handle = self._submit_task(p)
                    print("horovod::torch::_DistributedOptimizer::synchronize::loop1::p,handle,compress_status:",handle,compress_status)
                    self._handles[p] = handle

                for p, handle in self._handles.items():
                    if handle is None:
                        handle = self._submit_task(p)
                        print("horovod::torch::_DistributedOptimizer::synchronize::loop2::p,handle,ctx:",handle,compress_status)
                        self._handles[p] = handle

                finish_count = 0
                submit_count = len(self._handles)

                loss = None
                if closure is not None:
                    with torch.enable_grad():
                        loss = closure()

                while finish_count < submit_count:
                    try:
                        r = self._output_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
                    except queue.Empty:
                        continue

                    index, tensor, name, handle = r

                    # to process split tensor
                    if index >= 0:
                        original_name = self._rev_split_tensor_dict[name]
                        self._split_tensor_to_be_finish[original_name] -= 1
                        self._split_tensor_dict[original_name][index] = tensor

                        if self._split_tensor_to_be_finish[original_name] == 0:
                            tensor = torch.cat(self._split_tensor_dict[original_name], dim=0)
                            name = original_name
                        else:
                            continue

                    p = self._name_parameters.get(name)
                    if handle is not None:
                        tensor = synchronize(handle)

                    self._allreduce_delay[p] = self.backward_passes_per_step
                    p.grad.set_(tensor)

                    d_p = p.grad
                    weight_decay, momentum, dampening, nesterov, lr = self._sgd_parameter[p]
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf
                    p.data.add_(-lr, d_p)
                    finish_count += 1

                self._handles.clear()
                self._split_tensor_dict.clear()
                self._split_tensor_to_be_finish.clear()
                self._rev_split_tensor_dict.clear()
                self._synchronized = False
                return loss


            else:
                self.synchronize()
                self._synchronized = False
                return super(self.__class__, self).step(closure)
            
        else:
            self._synchronized = False
            return super(self.__class__, self).step(closure)
        

        
def DistributedOptimizer(optimizer, named_parameters=None,
                         compression=Compression.none,
                         backward_passes_per_step=1,
                         **kargs):
    """
    An optimizer that wraps another torch.optim.Optimizer, using an allreduce to
    average gradient values before applying gradients to model weights.

    Allreduce operations are executed after each gradient is computed by `loss.backward()`
    in parallel with each other. The `step()` method ensures that all allreduce operations are
    finished before applying gradients to the model.

    DistributedOptimizer exposes the `synchronize()` method, which forces allreduce operations
    to finish before continuing the execution. It's useful in conjunction with gradient
    clipping, or other operations that modify gradients in place before `step()` is executed.

    Example of gradient clipping:
    ```
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.synchronize()
    torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
    optimizer.step()
    ```

    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        named_parameters: A mapping between parameter names and values. Used for naming of
                          allreduce operations. Typically just `model.named_parameters()`.
        compression: Compression algorithm used during allreduce to reduce the amount
                     of data sent during the each parameter update step.  Defaults to
                     not using compression.
        backward_passes_per_step: Number of expected backward passes to perform
                                  before calling step()/synchronize(). This
                                  allows accumulating gradients over multiple
                                  mini-batches before executing averaging and
                                  applying them.
    """
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method with an allreduce implementation.
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_DistributedOptimizer.__dict__))
    return cls(optimizer.param_groups, named_parameters,
               compression, backward_passes_per_step, **kargs)


def broadcast_parameters(params, root_rank):
    """
    Broadcasts the parameters from root rank to all other processes.
    Typical usage is to broadcast the `model.state_dict()`,
    `model.named_parameters()`, or `model.parameters()`.

    Arguments:
        params: One of the following:
            - list of parameters to broadcast
            - dict of parameters to broadcast
        root_rank: The rank of the process from which parameters will be
                   broadcasted to all other processes.
    """
    if isinstance(params, dict):
        params = sorted(params.items())
    elif isinstance(params, list):
        # support both named_parameters() and regular parameters()
        params = [p if isinstance(p, tuple) else (None, p) for p in params]
    else:
        raise ValueError('invalid params of type: %s' % type(params))

    # Run asynchronous broadcasts.
    handles = []
    for name, p in params:
        handle = broadcast_async_(p, root_rank, name)
        handles.append(handle)

    # Wait for completion.
    for handle in handles:
        synchronize(handle)


def broadcast_optimizer_state(optimizer, root_rank):
    """
    Broadcasts an optimizer state from root rank to all other processes.

    Arguments:
        optimizer: An optimizer.
        root_rank: The rank of the process from which the optimizer will be
                   broadcasted to all other processes.
    """
    if isinstance(optimizer, torch.optim.LBFGS):
        # TODO(travis): L-BFGS cannot be easily supported without serializing
        # the entire state_dict, as its structure is deeply nested and contains
        # None type parameter values
        raise ValueError('cannot broadcast torch.optim.LBFGS state')

    state_dict = optimizer.state_dict()

    # Newly created optimizers will not have their state initialized, so
    # do that initialization here
    if len(state_dict['state']) == 0:
        for group in optimizer.param_groups:
            for p in group['params']:
                p.grad = p.data.new(p.size()).zero_()
        # This function accepts a torch.optim.Optimizer or a DistributedOptimizer
        # wrapped around a torch optimizer. Calling step() with a DistributedOptimizer
        # forces allreduce on all model parameters, which will result in deadlock
        # unless every rank calls step(). Therefore, to finish state initialization
        # only call optimizer.step() with a torch.optim.Optimizer.
        if optimizer.__module__ == DistributedOptimizer.__module__:
            super(optimizer.__class__, optimizer).step()
        else:
            optimizer.step()
        state_dict = optimizer.state_dict()

    # If the state_dict is still empty after initialization, then
    # the optimizer is stateless, and there is nothing to broadcast.
    # Furthermore, attempting to access the state dict would result in
    # an error.
    if len(state_dict['state']) == 0:
        return

    params = []
    callbacks = {}
    occurrences = collections.defaultdict(int)

    # Returns the full type structure of the possibly nested objects for recursive casting back
    def _get_types(x):
        if isinstance(x, collections.Iterable):
            return type(x), [_get_types(xi) for xi in x]
        else:
            return type(x)

    # Casts an object encoded in a tensor back into its original type and subtypes
    def _recursive_cast(x, dtype):
        if isinstance(dtype, tuple):
            t, dtypes = dtype
            x = t(x)
            return t([_recursive_cast(x[i], dtypes[i]) for i in range(len(x))])
        else:
            return dtype(x)

    # Some optimizer parameters may be represented as scalars instead of
    # tensors.  In such cases, we need to wrap the scalar in a tensor, then
    # broadcast, then update the appropriate value in the state_dict with the
    # new unwrapped scalar value via a callback.
    def _create_callback(pid, name, t, p):
        def _from_tensor():
            state_dict['state'][pid][name] = t(p.numpy()[0])
        return _from_tensor

    def _create_option_callback(index, option_key, option_tensor, dtypes):
        def _from_tensor():
            optimizer.param_groups[index][option_key] = _recursive_cast(option_tensor.numpy()[0], dtypes)
        return _from_tensor

    # Param groups are an ordered list, normally there is only one per model,
    # but users can add additional param groups for example to train
    # previously frozen layers
    for index, group in enumerate(state_dict['param_groups']):
        # Broadcast options like learning rate
        for option_key, option_value in group.items():
            if option_key == 'params':
                continue

            # Options like the learning rate are scalar, and need to be wrapped in tensors
            key = '%s.%d' % (option_key, index)
            dtypes = _get_types(option_value)
            option_tensor = torch.Tensor([option_value])
            callbacks[key] = _create_option_callback(index, option_key, option_tensor, dtypes)
            params.append((key, option_tensor))

        # The params list here is ordered by the layers in the model
        for pid in group['params']:
            param_state = state_dict['state'][pid]
            for name, p in param_state.items():
                # Some parameter names may appear more than once, in which
                # case we ensure they have a unique identifier defined by
                # their order
                occurrences[name] += 1
                key = '%s.%d' % (str(name), occurrences[name])

                if not torch.is_tensor(p):
                    # Wrap the scalar in a FloatTensor, and remember its type
                    # so we can cast it back after unwrapping
                    t = type(p)
                    p = torch.Tensor([p])
                    callbacks[key] = _create_callback(pid, name, t, p)

                params.append((key, p))

    # Synchronized broadcast of all parameters
    broadcast_parameters(params, root_rank)

    # Post-broadcast clenaup for non-tensor parameters
    for key, p in params:
        if key in callbacks:
            callbacks[key]()
