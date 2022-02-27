import torch

class _StorageBase():
    def __str__(self):
        content = ' ' + '\n '.join(str(self[i]) for i in torch._pyrange(len(self)))
        return 'a'

    def __repr__(self):
        return str(self)