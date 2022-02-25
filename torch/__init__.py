import torch._C

_tensor_classes = set()

from .Storage import _StorageBase

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
#
# a = DoubleStorage
# repr(a)