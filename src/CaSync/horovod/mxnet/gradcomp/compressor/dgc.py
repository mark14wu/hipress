import math
import mxnet as mx
from horovod.mxnet.mpi_ops import size
from . import SparseCompressor
from ..memory.dgc import DgcMemory


class DgcCompressor(SparseCompressor):
    def __init__(self, momentum=0.9, s_percent=0.001, sample_rate=0.001):
        super().__init__(DgcMemory(), s_percent)
        self.momentum = momentum
        self.sample_rate = sample_rate

    def memory_compensate(self, index, grad, state):
        pass

    def encode(self, index, grad, compressed_tensor):
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

    def _get_size_in_header(self, A):
        A = mx.nd.cast(A, dtype='int32')
        return A[0] + A[1]*256 + A[2]*65536 + A[3]*65536*256

    def get_compressed_size(self, tensor):
        return int((self._bits_of_compressed_grad(tensor.size) + 7) / 8)

    def get_server_compressed_size(self, tensor):
        return int((self._bits_of_compressed_grad(tensor.size * size()) + 7) / 8)
        # return self._get_size_in_header(tensor[:4])