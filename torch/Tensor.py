from . import TensorPrinting

class _TensorBase(object):

    def __repr__(self):
        return str(self)

    def __str__(self):
        return TensorPrinting.printTensor(self)