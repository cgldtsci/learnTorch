import math
import torch

def _printformat(storage):
    int_mode = True
    # TODO: use fmod?
    for value in storage:
        if value != math.ceil(value):
            int_mode = False
            break
    # tensor = torch.DoubleTensor(torch.DoubleStorage(storage.size()).copy_(storage)).abs()
    tensor = torch.DoubleTensor(torch.DoubleStorage(storage.size()).copy_(storage)).storage()
    return tensor

    # exp_min = tensor.min()
    # if exp_min != 0:
    #     exp_min = math.floor(math.log10(exp_min)) + 1
    # else:
    #     exp_min = 1
    # exp_max = tensor.max()
    # if exp_max != 0:
    #     exp_max = math.floor(math.log10(exp_max)) + 1
    # else:
    #     exp_max = 1
    #
    # scale = 1
    # exp_max = int(exp_max)
    # if int_mode:
    #     if exp_max > 9:
    #         format = '{:11.4e}'
    #         sz = 11
    #     else:
    #         sz = exp_max + 1
    #         format = '{:' + str(sz) + '.0f}'
    # else:
    #     if exp_max - exp_min > 4:
    #         sz = 11
    #         if abs(exp_max) > 99 or abs(exp_min) > 99:
    #             sz = sz + 1
    #         format = '{:' + str(sz) + '.4e}'
    #     else:
    #         if exp_max > 5 or exp_max < 0:
    #             sz = 7
    #             scale = math.pow(10, exp_max-1)
    #         else:
    #             if exp_max == 0:
    #                 sz = 7
    #             else:
    #                 sz = exp_max + 6
    #         format = '{:' + str(sz) + '.4f}'
    # return format, scale, sz


def _printVector(tensor):
    return  _printformat(tensor.storage())
    # fmt, scale, _ = _printformat(tensor.storage())
    # strt = ''
    # if scale != 1:
    #     strt += SCALE_FORMAT.format(scale)
    # return '\n'.join(fmt.format(val/scale) for val in tensor) + '\n'

def printTensor(self):
    if self.nDimension() == 0:
        return '[{} with no dimension]\n'.format(torch.typename(self))
    elif self.nDimension() == 1:
        strt = _printVector(self)
    return str(strt)
#     elif self.nDimension() == 2:
#         strt = _printMatrix(self)
#     else:
#         strt = _printTensor(self)
#
#     size_str = 'x'.join(str(size) for size in self.size())
#     strt += '[{} of size {}]\n'.format(torch.typename(self), size_str)
#     return '\n' + strt
