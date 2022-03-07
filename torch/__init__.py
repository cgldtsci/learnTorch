import torch._C

_tensor_classes = set()

# range gets shadowed by torch.range
def _pyrange(*args, **kwargs):
    return __builtins__['range'](*args, **kwargs)

def typename(o):
    return "." + o.__class__.__name__

from .Storage import _StorageBase
from .Tensor import _TensorBase

class DoubleStorage(torch._C.DoubleStorageBase, _StorageBase):
    pass
class FloatStorage(torch._C.FloatStorageBase, _StorageBase):
    pass
class LongStorage(torch._C.LongStorageBase, _StorageBase):
    pass
class IntStorage(torch._C.IntStorageBase, _StorageBase):
    pass
class ShortStorage(torch._C.ShortStorageBase, _StorageBase):
    pass
class CharStorage(torch._C.CharStorageBase, _StorageBase):
    pass
class ByteStorage(torch._C.ByteStorageBase, _StorageBase):
    pass

class DoubleTensor(torch._C.DoubleTensorBase, _TensorBase):
    pass
class FloatTensor(torch._C.FloatTensorBase, _TensorBase):
    pass
class LongTensor(torch._C.LongTensorBase, _TensorBase):
    pass
class IntTensor(torch._C.IntTensorBase, _TensorBase):
    pass
class ShortTensor(torch._C.ShortTensorBase, _TensorBase):
    pass
class CharTensor(torch._C.CharTensorBase, _TensorBase):
    pass
class ByteTensor(torch._C.ByteTensorBase, _TensorBase):
    pass

_C._initExtension()

# a = DoubleStorage
# repr(a)
