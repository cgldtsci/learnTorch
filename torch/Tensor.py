import torch
from . import TensorPrinting
import math
from functools import reduce

def _infer_sizes(sizes, total):
    to_infer = -1
    total_sizes = 1
    for i, size in enumerate(sizes):
        total_sizes *= size
        if size == -1:
            if to_infer >= 0:
                raise RuntimeError
            to_infer = i
    if to_infer >= 0:
        assert total % total_sizes == 0, "Can't make sizes have exactly %d elements" % total
        sizes[to_infer] = -total / total_sizes
    return sizes

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

    def typeAs(self, t):
        return self.type(t.type())

    def double(self):
        return self.type(torch.DoubleTensor)

    def float(self):
        return self.type(torch.FloatTensor)

    def long(self):
        return self.type(torch.LongTensor)

    def int(self):
        return self.type(torch.IntTensor)

    def short(self):
        return self.type(torch.ShortTensor)

    def char(self):
        return self.type(torch.CharTensor)

    def byte(self):
        return self.type(torch.ByteTensor)

    def copy_(self, other):
        torch._C._tensorCopy(self, other)
        return self

    def __deepcopy__(self, _memo):
        memo = _memo.setdefault('torch', {})
        if self._cdata in memo:
            return memo[self._cdata]
        new_storage = self.storage().__deepcopy__(_memo)
        new_tensor = self.new()
        new_tensor.set_(new_storage, self.storageOffset(), self.size(), self.stride())
        memo[self._cdata] = new_tensor
        return new_tensor

    def __reduce__(self):
        return type(self), (self.tolist(),)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return TensorPrinting.printTensor(self)

    def __iter__(self):
        return iter(map(lambda i: self.select(0, i), torch._pyrange(self.size(0))))

    def split(self, split_size, dim=0):
        dim_size = self.size(dim)
        num_splits = int(math.ceil(float(dim_size) / split_size))
        last_split_size = split_size - (split_size * num_splits - dim_size)
        def get_split_size(i):
            return split_size if i < num_splits-1 else last_split_size
        return [self.narrow(int(dim), int(i*split_size), int(get_split_size(i))) for i
                in torch._pyrange(0, num_splits)]

    def chunk(self, n_chunks, dim=0):
        split_size = math.ceil(float(self.size(dim)) / n_chunks)
        return self.split(split_size, dim)

    def tolist(self):
        dim = self.dim()
        if dim == 1:
            return [v for v in self]
        elif dim > 0:
            return [subt.tolist() for subt in self]
        return []


    def view(self, *args):
        dst = self.new()
        if len(args) == 1 and torch.isStorage(args[0]):
            sizes = args[0]
        else:
            sizes = torch.LongStorage(args)
        sizes = _infer_sizes(sizes, self.nElement())

        if reduce(lambda a,b: a * b, sizes) != self.nElement():
            raise RuntimeError('Invalid size for view. Input size: ' +
                    'x'.join(map(lambda v: str(v), self.size())) +
                    ', output size: ' +
                    'x'.join(map(lambda v: str(v), sizes)) + '.')

        assert self.isContiguous(), "expecting a contiguous tensor"
        dst.set_(self.storage(), self.storageOffset(), sizes)
        return dst

    def viewAs(self, tensor):
        return self.view(tensor.size())

    def permute(self, *args):
        perm = list(args)
        tensor = self
        n_dims = tensor.dim()
        assert len(perm) == n_dims, 'Invalid permutation'
        for i, p in enumerate(perm):
            if p != i and p != -1:
                j = i
                while True:
                    assert 0 <= perm[j] and perm[j] < n_dims, 'Invalid permutation'
                    tensor = tensor.transpose(j, perm[j])
                    perm[j], j = -1, perm[j]
                    if perm[j] == i:
                        break
                perm[j] = -1
        return tensor