import torch
from . import TensorPrinting

class _TensorBase(object):

    def __repr__(self):
        return str(self)

    def __str__(self):
        return TensorPrinting.printTensor(self)

    def __iter__(self):
        return iter(map(lambda i: self.select(0, i), torch._pyrange(self.size(0))))