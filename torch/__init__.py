_tensor_classes = set()

from .Storage import _StorageBase

class DoubleStorage(_StorageBase):
    pass

a = DoubleStorage
repr(a)