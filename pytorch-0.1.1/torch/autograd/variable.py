from .engine import ExecutionEngine

class Variable(object):

    _execution_engine = ExecutionEngine()

    # variable.data，即tensor里对应的某些attribute
    _fallthrough_methods = [
        'size',
        'stride',
        'nElement',
        'numel',
        'dim',
        # TODO: add more
    ]

    def __init__(self, tensor, creator=None, requires_grad=True):
        """variable对tensor进行初始化，如果无creator,则由Leaf来当其creator,Leaf的output就是Variable"""
        if creator is None:
            creator = Leaf(self, requires_grad)
        self.data = tensor
        self.creator = creator
        self._grad = None

    @property
    def grad(self):
        """若requires_grad
            self._grad 为None 则产生与variable 的tensor data 同长的 grad tensor,并初始化为0
            若不为None，则为self._grad
            """
        if self.creator.requires_grad:
            # TODO: this won't have to be zeroed in the future
            self._grad = self._grad or self.data.new(self.data.size()).zero_()
        return self._grad

    # 访问存在的属性时，会正常返回值，若该值不存在，则会进入最后的兜底函数__getattr__
    def __getattr__(self, name):
        if name in self._fallthrough_methods:
            return getattr(self.data, name)
        raise AttributeError(name)

    # 当实例对象通过[] 运算符取值时，会调用它的方法__getitem__
    # 这里会调用index function，且进行forward计算，会取得 当前variable的tensor，并返回tesnor[key],如key为int,则为对应的下标值
    def __getitem__(self, key):
        return Index(key)(self)[0]

    def backward(self, gradient=None):
        """variable反向求导，如无gradient则默认为1"""
        if gradient is None:
            if self.data.numel() != 1:
                raise RuntimeError('backward should be called only on a scalar (i.e. 1-element tensor) or with gradient w.r.t. the variable')
            gradient = self.data.new(1).fill_(1)
        self._execution_engine.run_backward(self, gradient)

    # variable 的repr 会调用tensor的repr
    def __repr__(self):
        return 'Variable containing:' + self.data.__repr__()

    def register_hook(self, name, hook):
        self.creator.register_hook(name, hook, self)

    def remove_hook(self, name):
        self.creator.remove_hook(name)

    def contiguous_(self):
        self.data = self.data.contiguous()
        return self

    # 类型转换，会复制
    def type(self, t):
        if t != type(self.data):
            return Copy(t)(self)[0]
        return self

    def add(self, other):
        if isinstance(other, Variable):
            return Add()(self, other)[0]
        else:
            return AddConstant(other)(self)[0]

    def sub(self, other):
        if isinstance(other, Variable):
            return Sub()(self, other)[0]
        else:
            return SubConstant(other)(self)[0]

    def mul(self, other):
        if isinstance(other, Variable):
            return Mul()(self, other)[0]
        else:
            return MulConstant(other)(self)[0]

    def div(self, other):
        if isinstance(other, Variable):
            return Div()(self, other)[0]
        else:
            return DivConstant(other)(self)[0]

    def pow(self, other):
        if isinstance(other, Variable):
            return Pow()(self, other)[0]
        else:
            return PowConstant(other)(self)[0]

    def view(self, *sizes):
        return View(*sizes)(self)[0]

    def t(self):
        return Transpose(0, 1)(self)[0]

    def transpose(self, dim1, dim2):
        return Transpose(dim1, dim2)(self)[0]

    def __add__(self, other):
        return self.add(other)
    __radd__ = __add__

    def __sub__(self, other):
        return self.sub(other)

    def __rsub__(self, other):
        return SubConstant(other, sub_tensor=True)(self)[0]

    def __mul__(self, other):
        return self.mul(other)
    __rmul__ = __mul__

    def __div__(self, other):
        return self.div(other)
    __truediv__ = __div__

    def __rdiv__(self, other):
        return DivConstant(other, div_by_tensor=True)(self)[0]
    __rtruediv__ = __rdiv__

    def __pow__(self, other):
        return self.pow(other)

    def __rpow__(self, other):
        return PowConstant(other, tensor_power=True)(self)[0]

    def __neg__(self):
        return Negate()(self)[0]


from .leaf import Leaf
from .functions import *
