from abc import ABC, abstractmethod
from ..memory import Memory
import mxnet as mx


class Compressor(ABC):
    def __init__(self, memory: Memory) -> None:
        self.memory = memory

    @abstractmethod
    def encode(self, index, grad, state, compressed_tensor):
        raise NotImplementedError()

    @abstractmethod
    def decode(self, compressed_tensor, output):
        raise NotImplementedError()

class SparseCompressor(Compressor):
    def __init__(self, memory: Memory) -> None:
        super().__init__(memory)
    
    @abstractmethod
    def _bits_of_compressed_grad(self, grad_size):
        raise NotImplementedError()
    
    @abstractmethod
    def server_encode(tensor, compressed_tensor):
        raise NotImplementedError()