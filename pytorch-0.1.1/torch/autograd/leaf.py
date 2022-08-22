from collections import OrderedDict
from .function import Function

class Leaf(Function):
    """
    当variable 的create不存在时，则由Leaf来当其creator
    Leaf的output为其对应的variable
    无input对应的前置函数
    """
    def __init__(self, variable, requires_grad):
        self.variable = variable
        self.output_ids = {id(variable): 0}
        self.previous_functions = []
        self.requires_grad = requires_grad
        self.backward_hooks = OrderedDict()

    def _do_forward(self, *input):
        raise NotImplementedError

    def _do_backward(self, *grad_output):
        """叶子点的backward，主要在于计算 variable的梯度累加"""
        assert len(grad_output) == 1
        # register_hook在其父类实现，values 为(hook,idx),所以这里可能代码有误
        for hook in self.backward_hooks.values():
            # 叶子节点并不会计算 grad_input
            hook(grad_output, grad_output)
        # variable 对应的梯度累加
        self.variable.grad.add_(grad_output[0])
        # 叶子返回值为空
        return tuple()
