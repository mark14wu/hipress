from abc import ABC, abstractmethod
from ..memory import Memory
import mxnet as mx


class Compressor(ABC):
    def __init__(self, memory: Memory) -> None:
        self.decode_params_map = dict()
        self.memory = memory
    
    @abstractmethod
    def memory_compensate(self, index, grad, state):
        raise NotImplementedError()

    @abstractmethod
    def encode(self, index, compressed_tensor):
        raise NotImplementedError()

    @abstractmethod
    def decode(self, compressed_tensor, output):
        raise NotImplementedError()

class SparseCompressor(Compressor):
    def __init__(self, memory: Memory, s_percent) -> None:
        super().__init__(memory)
        self.s_percent = s_percent