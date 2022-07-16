import mxnet as mx
from . import Memory
import math


class PowersgdMemory(Memory):
    def __init__(self) -> None:
        super().__init__()
        self._p_memory = dict()
        self._q_memory = dict()
        self._resd_gpus = dict()
        self._powersgd_M_gpus = dict()

    def initialize(self, index, grad, low_rank):
        if index in self._p_memory:
            return
        # view grad as 2-D matrix(M) with square shape
        interval = math.ceil(math.sqrt(grad.size))
        self._p_memory[index] = mx.nd.zeros(shape=(interval, low_rank), dtype=grad.dtype, ctx=grad.context)
        self._q_memory[index] = mx.nd.normal(shape=(interval, low_rank), dtype=grad.dtype, ctx=grad.context)
        self._resd_gpus[index] = mx.nd.zeros(shape=grad.size, dtype=grad.dtype, ctx=grad.context)
        self._powersgd_M_gpus[index] = mx.nd.zeros(shape=(interval, interval), dtype=grad.dtype, ctx=grad.context)
    
