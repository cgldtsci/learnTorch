#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Tensor.h"
#else

struct THPTensor {
  PyObject_HEAD
  THTensor *cdata;
};

bool THPTensor_(init)(PyObject *module);

#endif
