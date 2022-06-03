import mxnet as mx
import math
from horovod.mxnet.mpi_ops import size

from . import Compressor
from ..memory.topkadam import TopKAdamMemory
from . import SparseCompressor


class TopKAdamCompressor(SparseCompressor):
    def __init__(self, optimizer, s_percent=0.001, sample_rate=0.001):
        super().__init__(TopKAdamMemory())
        self.optimizer = optimizer
        self.s_percent = s_percent
        self.sample_rate = sample_rate
    
    def encode(self, index, grad, state, compressed_tensor):
        self.optimizer.update(index, self.memory.resd[index], grad, state)
        mx.nd.contrib.topk_adam_encode(
            data=self.memory.resd[index].reshape(grad.size),
            out=compressed_tensor,
            s_percent=self.s_percent,
            sample_rate=self.sample_rate,
        )
        mx.nd.zeros_like(grad, out=grad)

    def server_encode(self, tensor, compressed_tensor):
        mx.nd.contrib.topk_adam_server_encode(
            data=tensor,
            out=compressed_tensor
        )

    def decode(self, compressed_tensor, output):
        mx.nd.contrib.topk_adam_decode(
            data=compressed_tensor,
            is_add_to=1,
            out=output
        )

    def _bits_of_compressed_grad(self, grad_size):
        return (2 * math.ceil(grad_size * self.s_percent) + 1) * 32

    # def _get_size_in_header(self, A):
    #     A = mx.nd.cast(A, dtype='int32')
    #     return A[0] + A[1]*256 + A[2]*65536 + A[3]*65536*256

    # def get_compressed_size(self, tensor):
    #     return int((self._bits_of_compressed_grad(tensor.size) + 7) / 8)

    # def get_server_compressed_size(self, tensor):
    #     return self._get_size_in_header(tensor[:4])

    # def get_server_compressed_size(self, tensor):
        # bits = (2 * math.ceil(tensor.size * size() * self.s_percent) + 1) * 32
        # return int((bits + 7) / 8)