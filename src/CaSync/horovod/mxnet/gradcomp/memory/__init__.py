from abc import ABC, abstractmethod
import mxnet as mx


class Memory(ABC):
    @abstractmethod
    def initialize(self, index, grad):
        raise NotImplementedError()