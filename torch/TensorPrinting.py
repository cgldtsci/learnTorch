import torch

def printTensor(self):
    if self.nDimension() == 0:
        return '[{} with no dimension]\n'.format(torch.typename(self))
#     elif self.nDimension() == 1:
#         strt = _printVector(self)
#     elif self.nDimension() == 2:
#         strt = _printMatrix(self)
#     else:
#         strt = _printTensor(self)
#
#     size_str = 'x'.join(str(size) for size in self.size())
#     strt += '[{} of size {}]\n'.format(torch.typename(self), size_str)
#     return '\n' + strt