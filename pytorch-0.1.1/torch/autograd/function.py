from collections import OrderedDict
from .variable import Variable

class Function(object):

    def __init__(self):
        """
        previous_functions 前置函数列表为 各个input 所对应的(input_variable.creator,id(input_variable))
        outpu_ids 则为
        needs_input_grad: 输入变量是否需要grad的bool列表
        output_ids: 输出dict则为 id(output_variable):index
        """
        self.previous_functions = None
        self.output_ids = None
        self.needs_input_grad = None
        self.backward_hooks = OrderedDict()

    def __call__(self, *input):
        return self._do_forward(*input)

    def _do_forward(self, *input):
        """
        function的call方法调用该函数
        input为不定参数,类型为variable
        """
        # 获取input的data,为tensor类型
        unpacked_input = tuple(arg.data for arg in input)
        raw_output = self.forward(*unpacked_input)
        if not isinstance(raw_output, tuple):
            raw_output = (raw_output,)
        self.needs_input_grad = tuple(arg.creator.requires_grad for arg in input)
        # 如果由某个输入需要grad,则该func也是需要requires_grad
        self.requires_grad = any(self.needs_input_grad)
        # 将输出转化为varible ,且output 的creator为该 func
        output = tuple(Variable(tensor, self) for tensor in raw_output)

        # 前置函数列表为 各个input 所对应的(input_variable.creator,id(input_variable))
        self.previous_functions = [(arg.creator, id(arg)) for arg in input]
        # 输出dict则为 id(output_variable):index
        self.output_ids = {id(var): i for i, var in enumerate(output)}
        return output

    def _do_backward(self, grad_output):
        """输入类型为tensor"""
        grad_input = self.backward(grad_output)
        if not isinstance(grad_input, tuple):
            grad_input = (grad_input,)
        assert len(grad_input) == len(self.previous_functions), \
            self.__class__.__name__ + ' returned an invalid number of gradient tensors'

        # hook为自定义函数， idx为 output_variable所在的index
        for hook, idx in self.backward_hooks.values():
            # gi实际不起作用
            gi = grad_input if idx is None else grad_input[idx]
            # 执行hook函数
            hook(grad_input, grad_output)

        return grad_input

    def register_hook(self, name, hook, variable=None):
        """
        name为自定义的key名，如test，test2等
        hook为自定义函数
        variable不为空的话，则variable必须在 output_variable中
        """
        # 如果name存在,则不能再register
        assert name not in self.backward_hooks, \
            "Trying to register a second hook with name {}".format(name)
        variable_idx = self.output_ids[id(variable)] if variable else None
        self.backward_hooks[name] = (hook, variable_idx)

    def remove_hook(self, name):
        """删除hook"""
        assert name in self.backward_hooks, \
            "Trying to remove an inexistent hook with name {}".format(name)
        del self.backward_hooks[name]

    def forward(self, *input):
        """由继承类实现"""
        raise NotImplementedError

    def backward(self, *grad_output):
        "由继承类实现"
        raise NotImplementedError
