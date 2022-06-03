from . import Memory
import mxnet as mx


class DgcMemory(Memory):
    def __init__(self) -> None:
        self.u = dict()
        self.v = dict()
        self.memory_allocated = set()

    def initialize(self, index, tensor):
        if index in self.memory_allocated:
            return
        self.memory_allocated.add(index)
        self.u[index] = mx.nd.zeros_like(tensor).reshape(tensor.size)
        self.v[index] = mx.nd.zeros_like(tensor).reshape(tensor.size)