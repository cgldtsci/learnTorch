from tools.cwrap import cwrap
from tools.cwrap.plugins.THPPlugin import THPPlugin
from tools.cwrap.plugins.THPLongArgsPlugin import THPLongArgsPlugin
from tools.cwrap.plugins.ArgcountSortPlugin import ArgcountSortPlugin
from tools.cwrap.plugins.AutoGPU import AutoGPU

cwrap('torch/csrc/generic/TensorMethods.cwrap', plugins=[
    # THPLongArgsPlugin(),
    THPPlugin(),
    # ArgcountSortPlugin(),
    # AutoGPU()
])