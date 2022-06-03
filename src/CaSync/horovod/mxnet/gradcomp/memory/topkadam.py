import mxnet as mx
from . import Memory


class TopKAdamMemory(Memory):
    def __init__(self) -> None:
        self.resd = dict()
        self.memory_allocated = set()

    def initialize(self, index, grad):
        if index in self.memory_allocated:
            return
        self.memory_allocated.add(index)
        self.resd[index] = mx.nd.zeros_like(grad)
        # self.resd[index] = mx.nd.zeros(shape=grad.size, dtype=grad.dtype, ctx=grad.context)