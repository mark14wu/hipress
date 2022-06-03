import math
import mxnet as mx
from horovod.mxnet.mpi_ops import size
from . import SparseCompressor
from ..memory.dgc import DgcMemory


class DgcCompressor(SparseCompressor):
    def __init__(self, momentum=0.9, s_percent=0.001, sample_rate=0.001):
        super().__init__(DgcMemory())
        self.s_percent = s_percent
        self.momentum = momentum
        self.sample_rate = sample_rate

    def encode(self, index, grad, state, compressed_tensor):
        mx.nd.contrib.dgc_new(
            grad=grad.reshape(grad.size),
            u=self.memory.u[index],
            v=self.memory.v[index],
            out=compressed_tensor,
            s_percent=self.s_percent,
            sample_rate=self.sample_rate,
            momentum=self.momentum
        )

    def decode(self, compressed_tensor, output):
        mx.nd.contrib.dgc_new_r(
            data=compressed_tensor,
            is_add_to=1,
            out=output
        )

    def server_encode(self, tensor, compressed_tensor):
        mx.nd.contrib.dgc_new_server(
            data=tensor,
            out=compressed_tensor
        )

    def _bits_of_compressed_grad(self, grad_size):
        return (2 * math.ceil(grad_size * self.s_percent) + 1) * 32
