#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Tensor.h"
#else


struct THPTensor {
  PyObject_HEAD
  THTensor *cdata;
};

extern PyTypeObject THPTensorType;
extern PyObject *THPTensorClass;

bool THPTensor_(init)(PyObject *module);
bool THPTensor_(IsSubclass)(PyObject *tensor);

#endif
