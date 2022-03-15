import torch._C

_tensor_classes = set()
_storage_classes = set()

################################################################################
# Define basic utilities
################################################################################

def _import_dotted_name(name):
    components = name.split('.')
    obj = __import__(components[0])
    for component in components[1:]:
        obj = getattr(obj, component)
    return obj

def getDefaultTensorType():
    return _defaultTensorTypeName

# range gets shadowed by torch.range
def _pyrange(*args, **kwargs):
    return __builtins__['range'](*args, **kwargs)

def typename(o):
    # return o.__module__ + "." + o.__class__.__name__
    return "." + o.__class__.__name__

def isTensor(obj):
    return obj.__class__ in _tensor_classes

def isStorage(obj):
    return obj.__class__ in _storage_classes

def isLongStorage(obj):
    return isinstance(obj, LongStorage)

def setDefaultTensorType(t):
    global Tensor
    global Storage
    global _defaultTensorTypeName
    _defaultTensorTypeName = t
    Tensor = _import_dotted_name(t)
    Storage = _import_dotted_name(t.replace('Tensor', 'Storage'))

def getDefaultTensorType():
    return _defaultTensorTypeName

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


_storage_classes.add(DoubleStorage)
_storage_classes.add(FloatStorage)
_storage_classes.add(LongStorage)
_storage_classes.add(IntStorage)
_storage_classes.add(ShortStorage)
_storage_classes.add(CharStorage)
_storage_classes.add(ByteStorage)

_tensor_classes.add(DoubleTensor)
_tensor_classes.add(FloatTensor)
_tensor_classes.add(LongTensor)
_tensor_classes.add(IntTensor)
_tensor_classes.add(ShortTensor)
_tensor_classes.add(CharTensor)
_tensor_classes.add(ByteTensor)

# This shadows Torch.py and Storage.py
setDefaultTensorType('torch.DoubleTensor')

################################################################################
# Initialize extension
################################################################################

_C._initExtension()

#
# ################################################################################
# # Remove unnecessary members
# ################################################################################
#
# del DoubleStorageBase
# del FloatStorageBase
# del LongStorageBase
# del IntStorageBase
# del ShortStorageBase
# del CharStorageBase
# del ByteStorageBase
# del DoubleTensorBase
# del FloatTensorBase
# del LongTensorBase
# del IntTensorBase
# del ShortTensorBase
# del CharTensorBase
# del ByteTensorBase
