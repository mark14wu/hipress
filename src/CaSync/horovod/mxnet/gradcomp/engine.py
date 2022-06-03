from asyncio import gather
import mxnet as mx
from .compressor import Compressor, SparseCompressor
from .compressor.powersgd import PowerSGDCompressor
from horovod.mxnet.mpi_ops import gather_and_broadcast_, gather_and_broadcast
from horovod.mxnet.mpi_ops import size, rank, gather_, mpibroadcast_, allreduce_
import time


class GradientCompressEngine:
    def __init__(self, compressor: Compressor) -> None:
        self.compressor = compressor
        self.memory = self.compressor.memory
        self.buffer_initialized = set()
        self._comp_gpus = dict()
        self._comp_cpus = dict()

    def init_comm_buffer(self, index, grad):
        if index in self.buffer_initialized:
            return
        self.buffer_initialized.add(index)
        interval = grad.size
        self._comp_gpus[index] = mx.nd.zeros(shape=interval*5, dtype='uint8', ctx=grad.context)
        self._comp_cpus[index] = mx.nd.zeros(shape=interval*8, dtype='uint8', ctx=mx.context.cpu_pinned())
    
    def do_compressed_communication(self, index, grad, state):
        if isinstance(self.compressor, SparseCompressor):
            self._do_sparse_allgather(index, grad, state)
        elif isinstance(self.compressor, PowerSGDCompressor):
            self._do_powersgd_allreduce(index, grad)

    def _do_sparse_allgather(self, index, grad, state):
        self.init_comm_buffer(index, grad)
        self.memory.initialize(index, grad)

        # TODO: check constraints
        # if self._sparsity:
        #     assert self._compression_rate(size()) < 0.5, "compression rate too large! may use quantization."
        # assert self._bits_of_compressed_grad(grad.size) <= 8*self._comp_cpus[index[i]].size, "need more space for gathering!"

        rootid = index % size()
        compressed_size = int((self.compressor._bits_of_compressed_grad(grad.size)+7)/8) # ceil in uint8

        self.compressor.encode(index, grad, state, self._comp_gpus[index])

        # memcpy for gathering
        if rank() != rootid:
            self._comp_gpus[index][:compressed_size].copyto(self._comp_cpus[index][:compressed_size])
        else:
            root_start = rootid * compressed_size
            self._comp_gpus[index][:compressed_size].copyto(self._comp_cpus[index][root_start: root_start + compressed_size])
        gather_(self._comp_cpus[index], self._comp_cpus[index], grad, rootid, 
                num_elem=compressed_size)
        compressed_size_new = int(self.compressor._bits_of_compressed_grad(grad.size*size())/8)

        if rank() == rootid:
            for idx in range(size()):
                start = idx * compressed_size; offset = start + compressed_size
                self._comp_cpus[index][start:offset].copyto(self._comp_gpus[index][:compressed_size])
                self.compressor.decode(
                    compressed_tensor=self._comp_gpus[index][:compressed_size],
                    output=grad.reshape(grad.size)
                )
            self.compressor.server_encode(
                tensor=grad.reshape(grad.size),
                compressed_tensor=self._comp_gpus[index]
            )
            self._comp_gpus[index][:compressed_size_new].copyto(self._comp_cpus[index][:compressed_size_new])

        mpibroadcast_(self._comp_cpus[index], rootid, num_elem=compressed_size_new)

        if rank() != rootid:
            self._comp_cpus[index][:compressed_size_new].copyto(self._comp_gpus[index][:compressed_size_new])
            self.compressor.decode(
                compressed_tensor=self._comp_gpus[index][:compressed_size_new],
                output=grad.reshape(grad.size)
            )

        grad.__idiv__(size()) # average

    def init_powersgd_comm_buffer(self, index, grad):
        if index in self.buffer_initialized:
            return
        self.buffer_initialized.add(index)
        self._comp_gpus[index] = mx.nd.zeros(shape=grad.size, dtype=grad.dtype, ctx=grad.context)
        self._comp_cpus[index] = mx.nd.zeros(shape=grad.size, dtype=grad.dtype, ctx=mx.context.cpu_pinned())

    def _time_stamp(self, last_time, msg):
        if not self.profile:
            return
        if rank() == 0:
            print((msg + ":"), "%f ms" % (1000 * (time.time() - last_time)))
        return time.time()

    def _do_powersgd_allreduce(self, index, grad):
        self.memory.initialize(index, grad, self.compressor.decomp_rank)
        self.compressor.encode(index, grad)
        # allreduce P
        allreduce_(self.memory._p_memory[index], average=True)
        self.compressor.server_encode(index)
        # allreduce Q
        allreduce_(self.memory._q_memory[index], average=True)
        self.compressor.decode(index, grad)