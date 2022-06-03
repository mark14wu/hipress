from . import Compressor
from ..memory.powersgd import PowersgdMemory

import mxnet as mx


class PowerSGDCompressor(Compressor):
    def __init__(self, decomp_rank) -> None:
        super().__init__(PowersgdMemory())
        self.decomp_rank = decomp_rank

    def encode(self, index, grad):
        mx.nd.contrib.power_sgd_encode1(
            grad=grad,
            q=self.memory._q_memory[index],
            residual=self.memory._resd_gpus[index],
            m=self.memory._powersgd_M_gpus[index],
            out=self.memory._p_memory[index]
        )

    def server_encode(self, index):
        mx.nd.contrib.power_sgd_encode2(
            p=self.memory._p_memory[index],
            m=self.memory._powersgd_M_gpus[index],
            out=self.memory._q_memory[index]
        )

    def decode(self, index, grad):
        mx.nd.contrib.power_sgd_decode(
            grad=grad,
            q=self.memory._q_memory[index],
            residual=self.memory._resd_gpus[index],
            m=self.memory._powersgd_M_gpus[index],
            p=self.memory._p_memory[index]
        )
    
    
    