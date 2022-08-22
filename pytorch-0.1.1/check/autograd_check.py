from collections import OrderedDict

class ExecutionEngine(object):
    def __init__(self):
        pass

class Variable(object):

    _execution_engine = ExecutionEngine()

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



class Function(object):

    def __init__(self):
        self.previous_functions = None
        self.output_ids = None
        self.needs_input_grad = None
        self.backward_hooks = OrderedDict()

    def __call__(self, *input):
        return self._do_forward(*input)

class Leaf(Function):
    """
    当variable 的create不存在时，则由Leaf来当其creator
    Leaf的输出为其对应的variable、无前置函数
    """
    def __init__(self, variable, requires_grad):
        self.variable = variable
        self.output_ids = {id(variable): 0}
        self.previous_functions = []
        self.requires_grad = requires_grad
        self.backward_hooks = OrderedDict()


if __name__ == '__main__':
    print(0 or 2)


