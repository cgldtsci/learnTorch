import torch
from . import TensorPrinting

class _TensorBase(object):
    def new(self, *args, **kwargs):
        return self.__class__(*args, **kwargs)

    def type(self, t=None):
        if isinstance(t, str) or t is None:
            current = self.__module__ + '.' + self.__class__.__name__
            if t is None:
                return current
            if t == current:
                return self
            _, _, typename = t.partition('.')
            return torch._import_dotted_name(t)(self.size()).copy_(self)
        else:
            if t == type(self):
                return self
            return t(self.size()).copy_(self)

    def copy_(self, other):
        torch._C._tensorCopy(self, other)
        return self

    def __repr__(self):
        return str(self)

    def __deepcopy__(self, _memo):
        memo = _memo.setdefault('torch', {})
        if self._cdata in memo:
            return memo[self._cdata]
        new_storage = self.storage().__deepcopy__(_memo)
        new_tensor = self.new()
        new_tensor.set_(new_storage, self.storageOffset(), self.size(), self.stride())
        memo[self._cdata] = new_tensor
        return new_tensor

    def __str__(self):
        return TensorPrinting.printTensor(self)

    def __iter__(self):
        return iter(map(lambda i: self.select(0, i), torch._pyrange(self.size(0))))